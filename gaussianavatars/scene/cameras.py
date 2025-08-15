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
from torch import nn
import numpy as np


class Camera(nn.Module):
    def __init__(
        self, colmap_id, rt, intrinsics, bg, image_width, image, image_height, image_path,
        image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
        timestep=None, mask=None,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        if not isinstance(rt, torch.Tensor):
            rt = torch.tensor(rt).float()
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.tensor(intrinsics).float()
        self.rt = rt
        self.intrinsics = intrinsics
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.timestep = timestep
        if mask is not None:
            self.mask = torch.tensor(mask).bool()
        else:
            self.mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

    def get_center(self):
        return self.rt.inverse()[:3, 3]
