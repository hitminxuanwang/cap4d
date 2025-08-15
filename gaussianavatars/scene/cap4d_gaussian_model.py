# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import roma
from pytorch3d.io import load_obj

from cap4d.flame.flame import CAP4DFlameSkinner
from cap4d.mmdm.conditioning.mesh2img import VertexShader

from gaussianavatars.scene.net.positional_encoding import get_pos_enc
from gaussianavatars.scene.net.unet import define_G
from gaussianavatars.scene.gaussian_model import GaussianModel
from gaussianavatars.utils.mesh_utils import gen_uv_mesh
from gaussianavatars.utils.graphics_utils import compute_face_orientation
from gaussianavatars.utils.general_utils import get_expon_lr_func


FLAME_TEMPLATE_PATH = "data/assets/flame/cap4d_avatar_template.obj"
STD_DEFORM = 0.0108


class CAP4DGaussianModel(GaussianModel):
    def __init__(
        self, 
        model_params: Dict,
    ):
        super().__init__(model_params["sh_degree"])
        
        # Load FLAME skinner
        self.flame_model = CAP4DFlameSkinner(
            n_shape_params=150,
            n_expr_params=65,
            add_mouth=True,
            add_lower_jaw=model_params["use_lower_jaw"],
        ).cuda()

        # Load template mesh
        flame_verts, flame_faces, flame_aux = load_obj(FLAME_TEMPLATE_PATH)

        self.flame_verts = flame_verts.cuda()
        self.flame_faces_uvs = flame_faces.textures_idx.cuda()
        self.flame_faces = flame_faces.verts_idx.cuda()
        self.flame_uvs = flame_aux.verts_uvs.cuda()
        self.flame_uvs = self.flame_uvs * 2. - 1.
        self.flame_uvs[..., 1] = -self.flame_uvs[..., 1]

        self.flame_param = None
        self.static_neck = model_params["static_neck"]
        self.gaussian_init_type = model_params["gaussian_init_type"]
        self.n_gaussians_init = model_params["n_gaussians_init"]

        self.uv_resolution = model_params["uv_resolution"]
        self.n_points_per_triangle = model_params["n_points_per_triangle"]
        self.use_expr_mask = model_params["use_expr_mask"]

        n_pos_enc = 12
        self.pos_enc = get_pos_enc(n_pos_enc, self.uv_resolution).cuda()
        self.deform_net = define_G(
            3 + n_pos_enc * 2, 
            3, 
            64, 
            f'unet_{self.uv_resolution}', 
            n_layers=model_params["n_unet_layers"], 
            norm="instance"
        ).cuda()
        with torch.no_grad():
            # Initialize final deformation layer with zeros so that initial deformation is zero
            self.deform_net.model.model[-1].weight.data *= 0
            self.deform_net.model.model[-1].bias.data *= 0

        self.load_uv()

    @torch.no_grad()
    def load_uv(self):
        self.vert_shader = VertexShader().cuda()

        deformable_vertices = np.genfromtxt("data/assets/flame/deformable_verts.txt").astype(np.int64)
        vert_mask = torch.zeros_like(self.flame_verts[:, 0]).cuda()
        vert_mask[deformable_vertices] = 1
        deformable_face_mask = vert_mask[self.flame_faces]
        deformable_face_mask = deformable_face_mask.min(dim=-1)[0]

        # create pix_to_face map for UV rasterization and remeshing
        shader_input = {
            "positions": torch.cat([self.flame_uvs, torch.ones_like(self.flame_uvs[:, [1]])], dim=-1)[None],
        }
        _, fragments = self.vert_shader(
            shader_input, 
            self.flame_faces_uvs[None], 
            None, 
            None, 
            (self.uv_resolution, self.uv_resolution), 
            0.
        )
        self.fragments = fragments

        pix_to_face = self.fragments.pix_to_face
        uv_mask = pix_to_face >= 0

        self.uv_mask = uv_mask.permute(0, 3, 1, 2)

        pix_to_face[pix_to_face < 0] = 0

        deform_mask = deformable_face_mask[pix_to_face]
        deform_mask = torch.logical_and(deform_mask, uv_mask)
        
        self.deform_mask = deform_mask.permute(0, 3, 1, 2)

        uv_mask = uv_mask.permute(0, 3, 1, 2)
        self.uv_remesh_faces = gen_uv_mesh(uv_mask)

        # compute face area with template vertices
        # and count number of bindings
        template_verts = self.flame_verts.to(self.flame_faces.device)

        uv_remesh_verts = self.uv_remesh_flame_vertices(template_verts[None])[0]
        uv_remesh_verts = einops.rearrange(uv_remesh_verts, 'h w c -> (h w) c')

        triangles = uv_remesh_verts[self.uv_remesh_faces]

        ab = triangles[:, 1] - triangles[:, 0]
        ac = triangles[:, 2] - triangles[:, 0]
        face_area = 0.5 * torch.norm(torch.cross(ab, ac, dim=-1), dim=-1)

        gaussians_per_face = self.n_gaussians_init / face_area.sum() * face_area
        gaussians_per_face = gaussians_per_face.round().long().clamp(self.n_points_per_triangle)

        # adjust counts per triangle according to face area
        counts = []
        binding = []
        for i in range(gaussians_per_face.shape[0]):
            for j in range(gaussians_per_face[i]):
                counts.append(gaussians_per_face[i])
                binding.append(i)
        self.gaussian_counts = torch.tensor(counts).float().cuda()
        self.binding = torch.tensor(binding).to(torch.int64).cuda()
        self.binding_counter = gaussians_per_face.to(torch.int32)

    def load_meshes(self, train_meshes, test_meshes, tgt_meshes):
        meshes = train_meshes + test_meshes

        if len(tgt_meshes) > 0:
            meshes = meshes + tgt_meshes
            base_rot = tgt_meshes[0]['rot']
        else:
            base_rot = meshes[0]['rot']

        T = len(meshes)

        self.flame_param = {
            'shape': torch.from_numpy(meshes[0]['shape']),
            'base_rot': torch.from_numpy(base_rot), 
            'expr': torch.zeros([T, meshes[0]['expr'].shape[0]]),
            'eye_rot': torch.zeros([T, 3]),
            'rot': torch.zeros([T, 3]),
            'tra': torch.zeros([T, 3]),
        }

        if not self.static_neck:
            self.neck_rot_offset = nn.Embedding(
                T, 3, sparse=True, _weight=torch.zeros([T, 3])
            ).cuda()

        for i, mesh in enumerate(meshes):
            self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
            self.flame_param['eye_rot'][i] = torch.from_numpy(mesh['eye_rot'])
            self.flame_param['rot'][i] = torch.from_numpy(mesh['rot'])
            self.flame_param['tra'][i] = torch.from_numpy(mesh['tra'])
        
        for k, v in self.flame_param.items():
            self.flame_param[k] = v.float().cuda()

    def get_bbox_center(self):
        bbox_center = (self.verts.max(dim=1)[0] + self.verts.min(dim=1)[0]) / 2.
        return bbox_center

    def eval(self):
        self.deform_net.eval()

    def train(self):
        self.deform_net.train()

    def select_mesh_by_timestep(self, timestep):
        self.timestep = timestep
        
        base_rot = self.flame_param["base_rot"][None]
        curr_rot = self.flame_param["rot"][[timestep]]
        relative_rot = roma.rotvec_to_rotmat(curr_rot).inverse() @ roma.rotvec_to_rotmat(base_rot)
        relative_rot = roma.rotmat_to_rotvec(relative_rot)

        # limit neck rotation to not break the gaussians (hacky)
        MAX_NECK_ROT = 0.15
        relative_rot = torch.tanh(relative_rot / MAX_NECK_ROT) * MAX_NECK_ROT

        if not self.static_neck:
            # allow the neck to rotate during training to correct generated images
            neck_rot_offset = self.neck_rot_offset(
                torch.tensor([timestep], dtype=torch.long, device=relative_rot.device)
            )
            relative_rot = relative_rot + neck_rot_offset

        # compute flame for deformed and neutral mesh (with neck rotations)
        verts, _ = self.flame_model({
            "shape": self.flame_param["shape"],
            "expr": self.flame_param["expr"][[timestep]],
            "rot": self.flame_param["rot"][[timestep]],
            "tra": self.flame_param["tra"][[timestep]],
            "eye_rot": self.flame_param["eye_rot"][[timestep]],
            "neck_rot": relative_rot,
        })
        # convert from p3d to opencv convention
        verts[..., 1] = -verts[..., 1]
        verts[..., 2] = -verts[..., 2]

        neutral_verts, _ = self.flame_model({
            "shape": self.flame_param["shape"],
            "expr": self.flame_param["expr"][[timestep]] * 0.,
            "rot": self.flame_param["rot"][[timestep]],
            "tra": self.flame_param["tra"][[timestep]],
            "eye_rot": self.flame_param["eye_rot"][[timestep]] * 0.,
            "neck_rot": relative_rot,
        })
        # convert from p3d to opencv convention
        neutral_verts[..., 1] = -neutral_verts[..., 1]
        neutral_verts[..., 2] = -neutral_verts[..., 2]

        offsets = verts - neutral_verts

        self.update_mesh_properties(verts, offsets)

    def uv_remesh_flame_vertices(self, verts):
        verts_packed = verts[:, self.flame_faces]
        # remesh vertices
        uv_px_verts = self.vert_shader._rasterize_property(verts_packed, self.fragments)
        uv_px_verts = uv_px_verts.squeeze(3)

        return uv_px_verts

    def forward_unet(self, uv_offsets):
        if self.use_expr_mask:
            # import pdb; pdb.set_trace()
            # use mask to prevent deformations in undesired regions
            uv_offsets = uv_offsets * self.uv_mask

        deform_input = torch.cat([uv_offsets.detach(), self.pos_enc[None]], dim=1)
        nodeform_input = torch.cat([torch.zeros_like(uv_offsets), self.pos_enc[None]], dim=1)

        unet_input = torch.cat([deform_input, nodeform_input], dim=0)

        unet_output = self.deform_net(unet_input) * STD_DEFORM # unnormalization!

        deform_output, nodeform_output = unet_output.chunk(2, dim=0)

        # set deform mask places to neutral output so that it cannot deform
        deform_output = self.deform_mask * deform_output + torch.logical_not(self.deform_mask) * nodeform_output

        return deform_output, nodeform_output
    
    def update_mesh_properties(self, verts, offsets):        
        remeshed_verts = self.uv_remesh_flame_vertices(verts)
        remeshed_verts = einops.rearrange(remeshed_verts, 'b h w c -> b (h w) c')
        remeshed_offsets = self.uv_remesh_flame_vertices(offsets) / STD_DEFORM
        remeshed_offsets = einops.rearrange(remeshed_offsets, 'b h w c -> b c h w')
        
        deform_output, nodeform_output = self.forward_unet(remeshed_offsets)
        remeshed_deform = einops.rearrange(deform_output, 'b c h w -> b (h w) c')
        nodeform_offsets = einops.rearrange(nodeform_output, 'b c h w -> b (h w) c')

        self.deform_output = deform_output
        self.neutral_output = nodeform_output

        verts = remeshed_verts + remeshed_deform
        faces = self.uv_remesh_faces

        nodeform_verts = remeshed_verts + nodeform_offsets

        triangles = verts[:, faces]
        nodeform_triangles = nodeform_verts[:, faces]

        # neutral gaussian deformations
        nodeform_face_center = nodeform_triangles.mean(dim=-2).squeeze(0)
        # compute undeformed face orientation and scale (no U-Net deformation)
        nodeform_face_orien_mat, nodeform_face_scaling = compute_face_orientation(
            nodeform_verts.squeeze(0), 
            faces.squeeze(0), 
            return_scale=True
        )
        self.neutral_face_orien_mat = nodeform_face_orien_mat
        self.xyz_neutral = self.compute_face_xyz_transformed(nodeform_face_center, nodeform_face_orien_mat, nodeform_face_scaling)

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # compute deformed face orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0), 
            faces.squeeze(0), 
            return_scale=True
        )
        self.face_orien_quat = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces
    
    def compute_laplacian_loss(self):
        kernel = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]], device=self.deform_output.device,
        ).view(1, 1, 3, 3)

        b_ = self.deform_output.shape[0]
        deform = einops.rearrange(self.deform_output / STD_DEFORM, 'b c h w -> (b c) 1 h w')
        lap = F.conv2d(deform, kernel)
        lap = einops.rearrange(lap, '(b c) 1 h w -> b c h w', b=b_)
        lap = (lap ** 2).sum(dim=1, keepdim=True)
        self.laplacian = lap

        return lap.mean()

    def compute_neck_loss(self):
        if not self.static_neck:
            neck_rot_offset = self.neck_rot_offset(
                torch.tensor([self.timestep], dtype=torch.long, device=self.deform_output.device)
            )
            return neck_rot_offset.norm(dim=-1).mean()
        else:
            return 0.
    
    def print_neck_statistics(self):
        print(
            "mean:", self.neck_rot_offset.weight.mean(dim=0).detach(),
            "std:", self.neck_rot_offset.weight.std(dim=0).detach(),
        )
    
    def compute_relative_deformation_loss(self):
        # L2:
        diff = (((self.xyz_neutral - self.get_xyz) / STD_DEFORM) ** 2).sum(dim=1, keepdim=True)
        
        return diff.mean()
    
    def compute_relative_rotation_loss(self):
        # L2:
        relative_rot = self.neutral_face_orien_mat.inverse() @ self.face_orien_mat
            
        relative_rot = roma.rotmat_to_rotvec(relative_rot)

        diff = (relative_rot ** 2).sum(dim=-1)
        
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        # U-Net:
        self.optimizer.add_param_group(
            {
                'params': self.deform_net.parameters(), 
                'lr': training_args.deform_net_lr_init, 
                'weight_decay': training_args.deform_net_w_decay, 
                'name': "deform_net",
            }
        )

        self.deform_net_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deform_net_lr_init,
            lr_final=training_args.deform_net_lr_final,
            lr_delay_mult=training_args.deform_net_lr_delay_mult,
            max_steps=training_args.deform_net_lr_max_steps,
        )

        if not self.static_neck:
            self.neck_rot_offset.requires_grad = True
            self.neck_optimizer = torch.optim.SparseAdam([{
                    'params': self.neck_rot_offset.parameters(),
                    'lr': training_args.neck_lr_init,
                    'name': "neck_rot_offset",
                }], 
                lr=training_args.neck_lr_init, 
                eps=1e-18,
            )
            self.neck_scheduler_args = get_expon_lr_func(
                lr_init=training_args.neck_lr_init,
                lr_final=training_args.neck_lr_final,
                lr_delay_mult=training_args.neck_lr_delay_mult,
                max_steps=training_args.neck_lr_max_steps,
            )

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        if not self.static_neck:
            self.neck_optimizer.step()
            self.neck_optimizer.zero_grad(set_to_none = True)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_net":
                lr = self.deform_net_scheduler_args(iteration)
                param_group['lr'] = lr
            
        if not self.static_neck:
            for param_group in self.neck_optimizer.param_groups:
                if param_group["name"] == "neck_rot_offset":
                    lr = self.neck_scheduler_args(iteration)
                    param_group['lr'] = lr

        super().update_learning_rate(iteration)

    def capture(self):
        # save flame shape and base rotation for reenactment
        return {
            "shape": self.flame_param["shape"],
            "base_rot": self.flame_param["base_rot"],
            "deform_net": self.deform_net.state_dict(),
            "gaussians": super().capture(),
        }

    def restore(self, chkpt, training_args=None):
        self.flame_param["shape"] = chkpt["shape"]
        self.flame_param["base_rot"] = chkpt["base_rot"]
        self.deform_net.load_state_dict(chkpt["deform_net"])
        super().restore(chkpt["gaussians"], training_args)
