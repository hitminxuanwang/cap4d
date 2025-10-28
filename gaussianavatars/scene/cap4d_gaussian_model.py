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
import smplx
import os
import trimesh
import scipy.ndimage as ndimage
import cv2



FLAME_TEMPLATE_PATH = "data/assets/flame/cap4d_avatar_template.obj"
SMPL_TEMPLATE_PATH = "data/assets/smpl/smpl_template.obj"

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

        #print(flame_verts.shape, flame_faces.verts_idx.shape,flame_aux.verts_uvs.shape)

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


        output_dir = './' 
        os.makedirs(output_dir, exist_ok=True)
        remeshed_verts_np = uv_remesh_verts.cpu().numpy()
        remeshed_faces_np = self.uv_remesh_faces.cpu().numpy()
        remeshed_mesh = trimesh.Trimesh(vertices=remeshed_verts_np, faces=remeshed_faces_np)
        remeshed_mesh.export(os.path.join(output_dir, f'remeshed_flame_model_{self.uv_resolution}.obj'))

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
            if param_group["name"] == "deform_net" and not scene.gaussians.enable_deform_net:
                continue  
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

class SMPLGaussianModel(GaussianModel):
    def __init__(
        self, 
        model_params: Dict,
        smpl_model_path: str = 'data/assets/smpl/SMPL_NEUTRAL.pkl',
    ):
        super().__init__(model_params["sh_degree"])
        
        # Load SMPL model using smplx
        self.smpl_model = smplx.create(smpl_model_path, model_type='smpl', gender='neutral', use_pca=False).cuda()

        # Load template mesh for SMPL (assuming user provides an OBJ template similar to FLAME)
        smpl_verts, smpl_faces, smpl_aux = load_obj(SMPL_TEMPLATE_PATH)
        #debug

        #smpl_verts *= 20.0  # scale if needed
        #print(smpl_verts.shape,smpl_faces.verts_idx.shape, smpl_aux.verts_uvs.shape)

        self.smpl_verts = smpl_verts.cuda()
        self.smpl_faces_uvs = smpl_faces.textures_idx.cuda()
        self.smpl_faces = smpl_faces.verts_idx.cuda()


        self.smpl_uvs = smpl_aux.verts_uvs.cuda()
        self.smpl_uvs = self.smpl_uvs * 2. - 1.
        self.smpl_uvs[..., 1] = -self.smpl_uvs[..., 1]



        self.smpl_param = None
        self.static_neck = model_params.get("static_neck", True)  # Reuse if applicable
        self.gaussian_init_type = model_params["gaussian_init_type"]
        self.n_gaussians_init = model_params["n_gaussians_init"]

        self.uv_resolution = 256 #model_params["uv_resolution"]
        self.n_points_per_triangle = model_params["n_points_per_triangle"]
        self.use_expr_mask = model_params.get("use_expr_mask", False)  # Optional for SMPL


        self.enable_deform_net = True


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

        # For SMPL, define deformable vertices (assume all or load from file similar to FLAME)
        deformable_vertices = np.genfromtxt("data/assets/smpl/deformable_verts.txt").astype(np.int64)  # Assume user provides this
        vert_mask = torch.zeros_like(self.smpl_verts[:, 0]).cuda()
        vert_mask[deformable_vertices] = 1
        deformable_face_mask = vert_mask[self.smpl_faces]
        deformable_face_mask = deformable_face_mask.min(dim=-1)[0]

        # print(self.smpl_uvs.shape)

        # res = 1024
        # img = np.ones((res, res, 3), dtype=np.uint8) * 255

        # uvs = self.smpl_uvs.detach().cpu().numpy().copy()
        # faces_uv = self.smpl_faces_uvs.detach().cpu().numpy().copy()  

        # uvs = (uvs + 1.0) / 2.0

        # uvs[:, 0] = uvs[:, 0] * (res - 1)
        # uvs[:, 1] = (1 - uvs[:, 1]) * (res - 1)

        # for tri in faces_uv:
        #     pts = uvs[tri].astype(np.int32)
        #     cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)
        #     cv2.fillPoly(img, [pts], color=(200, 200, 255))

        # cv2.imwrite("uv_layout.png", img)
        # print("Saved uv_layout.png")


        # create pix_to_face map for UV rasterization and remeshing
        shader_input = {
            "positions": torch.cat([self.smpl_uvs, torch.ones_like(self.smpl_uvs[:, [1]])], dim=-1)[None],
        }
        _, fragments = self.vert_shader(
            shader_input, 
            self.smpl_faces_uvs[None], 
            None, 
            None, 
            (self.uv_resolution, self.uv_resolution), 
            0.
        )
        self.fragments = fragments
        #print(self.fragments.pix_to_face)

        pix_to_face = self.fragments.pix_to_face
        uv_mask = pix_to_face >= 0


        # face_index = pix_to_face[0, :, :, 0].cpu().numpy()
        # face_index = cv2.normalize(face_index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # face_index = cv2.applyColorMap(face_index, cv2.COLORMAP_JET)  
        # cv2.imwrite('face_index.png', face_index)


        # uv_mask_img = uv_mask[0, :, :, 0].cpu().numpy().astype(np.uint8) * 255
        # cv2.imwrite('uv_mask.png', uv_mask_img)


        # print(pix_to_face.shape, uv_mask.shape)

        self.uv_mask = uv_mask.permute(0, 3, 1, 2)

        pix_to_face[pix_to_face < 0] = 0

        deform_mask = deformable_face_mask[pix_to_face]
        deform_mask = torch.logical_and(deform_mask, uv_mask)
        
        self.deform_mask = deform_mask.permute(0, 3, 1, 2)

        uv_mask = uv_mask.permute(0, 3, 1, 2)
        self.uv_remesh_faces = gen_uv_mesh(uv_mask)

        # compute face area with template vertices
        # and count number of bindings
        template_verts = self.smpl_verts.to(self.smpl_faces.device)

        uv_remesh_verts = self.uv_remesh_smpl_vertices(template_verts[None])[0]  # Adjusted method name
        uv_remesh_verts = einops.rearrange(uv_remesh_verts, 'h w c -> (h w) c')

        # for debug
        output_dir = './' 
        os.makedirs(output_dir, exist_ok=True)
        remeshed_verts_np = uv_remesh_verts.cpu().numpy()
        remeshed_faces_np = self.uv_remesh_faces.cpu().numpy()
        remeshed_mesh = trimesh.Trimesh(vertices=remeshed_verts_np, faces=remeshed_faces_np)
        remeshed_mesh.export(os.path.join(output_dir, f'remeshed_model_{self.uv_resolution}.obj'))

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

        #print(len(self.gaussian_counts), len(self.binding), len(self.binding_counter))
        #print(self.gaussian_counts, self.binding, self.binding_counter)

    def load_meshes(self, train_meshes, test_meshes, tgt_meshes):
        meshes = train_meshes + test_meshes

        if len(tgt_meshes) > 0:
            meshes = meshes + tgt_meshes
            base_rot = tgt_meshes[0].get('rot', np.zeros(3))  # Adjust for SMPL
        else:
            base_rot = meshes[0].get('rot', np.zeros(3))

        T = len(meshes)

        self.smpl_param = {
            'betas': torch.from_numpy(meshes[0].get('betas', np.zeros(10))),
            'base_rot': torch.from_numpy(base_rot), 
            'body_pose': torch.zeros([T, 69]),  # SMPL body pose
            'global_orient': torch.zeros([T, 3]),
            'tra': torch.zeros([T, 3]),
        }

        if not self.static_neck:
            self.neck_rot_offset = nn.Embedding(
                T, 3, sparse=True, _weight=torch.zeros([T, 3])
            ).cuda()

        for i, mesh in enumerate(meshes):
            self.smpl_param['body_pose'][i] = torch.from_numpy(mesh.get('body_pose', np.zeros(69)))
            self.smpl_param['global_orient'][i] = torch.from_numpy(mesh.get('global_orient', np.zeros(3)))
            #self.smpl_param['tra'][i] = torch.from_numpy(mesh.get('tra', np.zeros(3)))
            tra_value = mesh.get('tra', np.zeros(3))
            if isinstance(tra_value, torch.Tensor):
                tra_value = tra_value.cpu().numpy()  # Convert Tensor to NumPy
            self.smpl_param['tra'][i] = torch.from_numpy(tra_value)


        for k, v in self.smpl_param.items():
            self.smpl_param[k] = v.float().cuda()

    def get_bbox_center(self):
        bbox_center = (self.verts.max(dim=1)[0] + self.verts.min(dim=1)[0]) / 2.
        return bbox_center

    def eval(self):
        self.deform_net.eval()

    def train(self):
        self.deform_net.train()

    def select_mesh_by_timestep(self, timestep):
        self.timestep = timestep
        
        base_rot = self.smpl_param["base_rot"][None]
        curr_rot = self.smpl_param["global_orient"][[timestep]]  # Use global_orient for rotation in SMPL
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

        body_pose = self.smpl_param['body_pose'][timestep].unsqueeze(0)
        global_orient = self.smpl_param['global_orient'][timestep].unsqueeze(0)

        # print("Timestep", timestep)
        # print("SMPL Size",self.smpl_param['tra'].shape)
        # print("Body Pose",body_pose)
        # print("Global Orient",global_orient)

        transl = self.smpl_param['tra'][timestep].unsqueeze(0)

        # compute SMPL for deformed and neutral mesh (with neck rotations)
        smpl_output = self.smpl_model(
            betas=self.smpl_param["betas"][None],
            body_pose=body_pose,
            global_orient=global_orient ,  # Adjust global_orient with neck
            transl=None,
        )
        #verts = smpl_output.vertices.squeeze(0)
        verts = smpl_output.vertices
        # convert from p3d to opencv convention if needed (adjust based on coord system)
        # verts[..., 1] = -verts[..., 1]
        # verts[..., 2] = -verts[..., 2]

        # print("First vertex coordinate in Cap4D (deformed):", verts[0][0].cpu().numpy() if verts.is_cuda else verts[0][0].numpy())
        

        # vertices_center = verts[0].mean(dim=0).cpu().numpy() if verts.is_cuda else verts[0].mean(dim=0).numpy()
        # print("Vertices center in Cap4D:", vertices_center)

        # vertices_min = verts[0].min(dim=0).values.cpu().numpy() if verts.is_cuda else verts[0].min(dim=0).values.numpy()
        # vertices_max = verts[0].max(dim=0).values.cpu().numpy() if verts.is_cuda else verts[0].max(dim=0).values.numpy()
        # xyz_extend = vertices_max - vertices_min
        # print("XYZ extend in Cap4D:", xyz_extend)



        neutral_output = self.smpl_model(
            betas=self.smpl_param["betas"][None],
            body_pose=torch.zeros_like(self.smpl_param["body_pose"][[timestep]]),
            global_orient=self.smpl_param["global_orient"][[timestep]],
            transl=self.smpl_param["tra"][[timestep]],
        )
        #neutral_verts = neutral_output.vertices.squeeze(0)
        neutral_verts = neutral_output.vertices
        #neutral_verts[..., 1] = -neutral_verts[..., 1]
        #neutral_verts[..., 2] = -neutral_verts[..., 2]


        #print("First vertex coordinate in Cap4D (deformed):", verts[0][0].cpu().numpy() if verts.is_cuda else verts[0][0].numpy())
        

        # vertices_center = neutral_verts[0].mean(dim=0).cpu().numpy() if neutral_verts.is_cuda else neutral_verts[0].mean(dim=0).numpy()
        # print("Neural Vertices center in Cap4D:", vertices_center)

        # vertices_min = neutral_verts[0].min(dim=0).values.cpu().numpy() if neutral_verts.is_cuda else neutral_verts[0].min(dim=0).values.numpy()
        # vertices_max = neutral_verts[0].max(dim=0).values.cpu().numpy() if neutral_verts.is_cuda else neutral_verts[0].max(dim=0).values.numpy()
        # xyz_extend = vertices_max - vertices_min
        # print("Neural XYZ extend in Cap4D:", xyz_extend)


        offsets = verts - neutral_verts

        self.update_mesh_properties(verts, offsets)

    def uv_remesh_smpl_vertices(self, verts):
        verts_packed = verts[:, self.smpl_faces]

        # remesh vertices
        uv_px_verts = self.vert_shader._rasterize_property(verts_packed, self.fragments)
        uv_px_verts = uv_px_verts.squeeze(3)

        return uv_px_verts

    def forward_unet(self, uv_offsets):
        if self.use_expr_mask:
            uv_offsets = uv_offsets * self.uv_mask

        deform_input = torch.cat([uv_offsets.detach(), self.pos_enc[None]], dim=1)
        nodeform_input = torch.cat([torch.zeros_like(uv_offsets), self.pos_enc[None]], dim=1)

        unet_input = torch.cat([deform_input, nodeform_input], dim=0)

        unet_output = self.deform_net(unet_input) * STD_DEFORM  # unnormalization!

        deform_output, nodeform_output = unet_output.chunk(2, dim=0)

        # set deform mask places to neutral output so that it cannot deform
        deform_output = self.deform_mask * deform_output + torch.logical_not(self.deform_mask) * nodeform_output

        return deform_output, nodeform_output
    
    def update_mesh_properties(self, verts, offsets):        
        remeshed_verts = self.uv_remesh_smpl_vertices(verts)
        remeshed_verts = einops.rearrange(remeshed_verts, 'b h w c -> b (h w) c')
        remeshed_offsets = self.uv_remesh_smpl_vertices(offsets) / STD_DEFORM
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
        

        # vertices_min = verts[0].cpu().min(dim=0).values.cpu().numpy() if verts.is_cuda else verts[0].min(dim=0).values.numpy()
        # vertices_max = verts[0].cpu().max(dim=0).values.cpu().numpy() if verts.is_cuda else verts[0].max(dim=0).values.numpy()
        # xyz_extend = vertices_max - vertices_min
        # print("Final XYZ extend in Cap4D:", xyz_extend)
        #debug
        #self.verts += torch.tensor([0.0, 0.0, 5.0], device=self.verts.device)

    
    def compute_laplacian_loss(self):
        
        if not self.enable_deform_net:
            return 0.0

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
        if not self.enable_deform_net:
            return 0.0
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
        if not self.enable_deform_net:
            return 0.0
        # L2:
        diff = (((self.xyz_neutral - self.get_xyz) / STD_DEFORM) ** 2).sum(dim=1, keepdim=True)
        
        return diff.mean()
    
    def compute_relative_rotation_loss(self):
        if not self.enable_deform_net:
            return 0.0
        # L2:
        relative_rot = self.neutral_face_orien_mat.inverse() @ self.face_orien_mat
            
        relative_rot = roma.rotmat_to_rotvec(relative_rot)

        diff = (relative_rot ** 2).sum(dim=-1)
        
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        if not self.enable_deform_net:
            return None
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

        # if not self.static_neck:
        #     self.neck_rot_offset.requires_grad = True
        #     self.neck_optimizer = torch.optim.SparseAdam([{
        #             'params': self.neck_rot_offset.parameters(),
        #             'lr': training_args.neck_lr_init,
        #             'name': "neck_rot_offset",
        #         }], 
        #         lr=training_args.neck_lr_init, 
        #         eps=1e-18,
        #     )
        #     self.neck_scheduler_args = get_expon_lr_func(
        #         lr_init=training_args.neck_lr_init,
        #         lr_final=training_args.neck_lr_final,
        #         lr_delay_mult=training_args.neck_lr_delay_mult,
        #         max_steps=training_args.neck_lr_max_steps,
        #     )

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        # if not self.static_neck:
        #     self.neck_optimizer.step()
        #     self.neck_optimizer.zero_grad(set_to_none = True)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform_net":
                lr = self.deform_net_scheduler_args(iteration)
                param_group['lr'] = lr
            
        # if not self.static_neck:
        #     for param_group in self.neck_optimizer.param_groups:
        #         if param_group["name"] == "neck_rot_offset":
        #             lr = self.neck_scheduler_args(iteration)
        #             param_group['lr'] = lr

        super().update_learning_rate(iteration)

    def capture(self):
        # save smpl betas and base rotation for reenactment (adjusted from flame)
        return {
            "betas": self.smpl_param["betas"],
            "base_rot": self.smpl_param["base_rot"],
            "deform_net": self.deform_net.state_dict(),
            "gaussians": super().capture(),
        }

    def restore(self, chkpt, training_args=None):
        self.smpl_param["betas"] = chkpt["betas"]
        self.smpl_param["base_rot"] = chkpt["base_rot"]
        self.deform_net.load_state_dict(chkpt["deform_net"])
        super().restore(chkpt["gaussians"], training_args)

    def load_smpl_params_from_npz(self, npz_path: str, timestep: int):
        data = np.load(npz_path)
        betas = torch.from_numpy(data['betas']).float().cuda()
        body_pose = torch.from_numpy(data['body_pose']).float().cuda()
        global_orient = torch.from_numpy(data['global_orient']).float().cuda()
        smpl_output = self.smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient)
        self.smpl_verts = smpl_output.vertices.cuda()  
        self.smpl_faces = self.smpl_model.faces_tensor.cuda()  