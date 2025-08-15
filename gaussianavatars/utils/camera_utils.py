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

from tqdm import tqdm
import numpy as np

from gaussianavatars.scene.cameras import Camera

WARNED = False

def loadCam(id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    assert resolution_scale == 1.

    return Camera(
        colmap_id=cam_info.uid, 
        rt=cam_info.rt, 
        intrinsics=cam_info.intrinsics, 
        image_width=orig_w, 
        image_height=orig_h,
        bg=cam_info.bg, 
        image=cam_info.image, 
        image_path=cam_info.image_path,
        image_name=cam_info.image_name, 
        uid=id, 
        timestep=cam_info.timestep, 
        mask=cam_info.mask,
    )

def cameraList_from_camInfos(cam_infos, resolution_scale):
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        camera_list.append(loadCam(id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt = camera.rt

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'intrinsics' : camera.intrinsics.tolist(),
    }
    return camera_entry
