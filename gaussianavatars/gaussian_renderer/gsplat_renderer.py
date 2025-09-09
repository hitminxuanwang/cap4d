#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from gsplat.rendering import rasterization
from gaussianavatars.scene.gaussian_model import GaussianModel
import roma
import numpy as np


def render(
    viewpoint_camera, 
    pc : GaussianModel, 
    bg_color : torch.Tensor, 
    clip=False,
    compute_depth=False,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    scales = pc.get_scaling
    rotations = pc.get_rotation
    xyz = pc.get_xyz

    view_rt = viewpoint_camera.rt.cuda()

    clip_distance = 1e3
    if clip:
        gaussian_center = pc.get_bbox_center()[0]
        cam_center = view_rt.inverse()[:3, 3].cuda()
        clip_distance = (gaussian_center - cam_center).norm() + 0.025

    rendered_image, alphas, meta = rasterization(
        means=xyz,
        quats=rotations,
        scales=scales,
        opacities=pc.get_opacity[..., 0],
        colors=pc.get_features,
        viewmats=view_rt[None],
        Ks=viewpoint_camera.intrinsics[None].cuda(),
        width=viewpoint_camera.image_width,
        height=viewpoint_camera.image_height,
        sh_degree=pc.active_sh_degree,
        backgrounds=bg_color[None],
        packed=False,
        far_plane=clip_distance,
        render_mode="RGB" if not compute_depth else "RGB+ED",
    )
    
    rgb_image = rendered_image[0].permute(2, 0, 1)
    depth_image = None
    if compute_depth:
        depth_image = rgb_image[[3]]
        rgb_image = rgb_image[:3]

    if meta["means2d"].requires_grad:
        meta["means2d"].retain_grad()

    import pdb; pdb.set_trace()
    # making this compatible with older versions of gsplat where ndim of radii = 2
    if meta["radii"].ndim == 3:
        visibility_filter = (meta["radii"][0] > 0).all(dim=-1)
        radii = meta["radii"][0].float().norm(dim=-1)
    else:
        visibility_filter = meta["radii"][0] > 0
        radii = meta["radii"][0]

    return {
        "render": rgb_image,
        "alpha": alphas[0].permute(2, 0, 1),
        "viewspace_points": meta["means2d"],
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": depth_image,
    }


def export_gaussians(pc: GaussianModel):
    xyz = pc.get_xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = pc._opacity.detach().cpu().numpy()
    scale = pc.scaling_inverse_activation(pc.get_scaling).detach().cpu().numpy()
    # scale = pc.get_scaling.detach().cpu().numpy()
    rotation = pc.get_rotation.detach().cpu().numpy()

    return {
        "xyz": xyz,
        "normals": normals,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacities": opacities,
        "scale": scale,
        "rotation": rotation,
    }
