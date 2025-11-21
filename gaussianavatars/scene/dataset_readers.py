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

from typing import NamedTuple, Optional, Dict, Any
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

from flowface.flame.utils import batch_rodrigues, OPENCV2PYTORCH3D

from cap4d.datasets.utils import (
    adjust_intrinsics_crop,
    get_crop_mask,
)


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: Optional[str]
    val_cameras: list = []
    train_meshes: dict = {}
    test_meshes: dict = {}
    tgt_meshes: dict = {}
    tgt_cameras: list = []


class CVCameraInfo(NamedTuple):
    uid: int
    rt: np.array
    intrinsics: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array
    bg: np.array = np.array([1, 1, 1])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None


def reverse_transform(extr, rot, tra):
    """
    Adjust extrinsics and head rotation to fix head at origin.
    This means that the camera rotates around head instead of head rotating in world coords.
    We need this to get head rotation dependent lighting.
    This is a hack though, technically view and head pose changes will lead to the same lighting effects
    - it looks cool though :)
    """
    T_head = torch.eye(4)[None]
    T_head[:, :3, :3] = batch_rodrigues(torch.tensor(rot)[None])
    T_head[:, :3, 3] = torch.tensor(tra)
    new_extr = torch.tensor(extr).float() @ OPENCV2PYTORCH3D @ T_head[0] @ OPENCV2PYTORCH3D.inverse()
    # Since we rotate camera around head we need to set rot and tra to zero
    new_rot = rot * 0.
    new_tra = tra * 0.

    return new_extr, new_rot, new_tra


def loadCAP4DItem(idx, flame_path, image_path):
    flame_item = dict(np.load(flame_path))

    # we are loading cropped images
    with Image.open(image_path) as img_file:
        image = img_file.copy()

    bg = np.array([1, 1, 1])

    orig_resolution = flame_item["resolutions"][0]
    crop_width, crop_height = image.size
    crop_box = flame_item["crop_box"]

    # adjust intrinsics according to crop box
    fx, fy, cx, cy = [flame_item[key][0, 0] for key in ["fx", "fy", "cx", "cy"]]
    fx, fy, cx, cy = adjust_intrinsics_crop(fx, fy, cx, cy, crop_box, crop_width)

    # if the image is cropped, get outcropping mask
    crop_mask = get_crop_mask(orig_resolution, crop_width, crop_box)

    extr, rot, tra = reverse_transform(
        flame_item["extr"][0],
        flame_item["rot"][0],
        flame_item["tra"][0],
    )

    intrinsics = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
    )

    flame_out = {
        "shape": flame_item["shape"],
        "expr": flame_item["expr"][0],
        "eye_rot": flame_item["eye_rot"][0],
        "rot": rot,
        "tra": tra,
    }

    cam_info = CVCameraInfo(
        rt=extr,
        intrinsics=intrinsics,
        uid=idx, 
        bg=bg, 
        image=image, 
        image_path=image_path, 
        image_name=image_path.stem, 
        width=crop_width, 
        height=crop_height, 
        timestep=idx, 
        camera_id=idx,
        mask=crop_mask,
    )

    return cam_info, flame_out



def convert_cam_to_3d_trans(cams, weight=1.4):
    # Handle both 1D and 2D inputs
    original_ndim = cams.ndim
    if not isinstance(cams, torch.Tensor):
        cams = torch.from_numpy(cams)
    
    if cams.ndim == 1:
        cams = cams.unsqueeze(0)  # Use unsqueeze instead of reshape for clarity
    
    s = cams[:, 0]
    tx = cams[:, 1]
    ty = cams[:, 2]
    depth = 1. / s
    dx = tx / s
    dy = ty / s
    trans3d = torch.stack([dx, dy, depth], dim=1) * weight
    
    # If original was 1D, squeeze back to (3,)
    if original_ndim == 1:
        trans3d = trans3d.squeeze(0)
    
    return trans3d


def loadSMPLItem(idx, smpl_path, image_path):
    smpl_item = dict(np.load(smpl_path))

    # we are loading cropped images
    with Image.open(image_path) as img_file:
        image = img_file.copy()

    crop_width, crop_height = image.size

    #print(image.size)

    bg = np.array([1, 1, 1])

    orig_resolution = np.array([crop_height, crop_width])

    crop_box = None

    # Load intrinsics directly from NPZ (new export format)
    fx = smpl_item["fx"]
    fy = smpl_item["fy"]
    cx = smpl_item["cx"]
    cy = smpl_item["cy"]

    # if the image is cropped, get outcropping mask (unchanged)
    crop_mask = np.ones((crop_height, crop_width), dtype=bool)

    # Load extrinsics from R and T (new export format)
    rot = smpl_item["R"]  # 3x3 rotation matrix
    tra = smpl_item["T"].squeeze()  # 3x1 translation vector, flattened to 3-element array

    extr = np.eye(4)
    extr[:3, :3] = rot  # Set rotation part
    extr[:3, 3] = tra   # Set translation part


    # z_flip = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # extr[:3, :3] = z_flip @ extr[:3, :3]

    # extr[:3, 3] = (z_flip @ extr[:3, 3].reshape(3, 1)).squeeze()

    intrinsics = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
    )

    global_orient = smpl_item["global_orient"]  # (3,) axis-angle vector
    body_pose = smpl_item["body_pose"]          # (69,) axis-angle vector
    thetas = np.concatenate([global_orient, body_pose])  # Full 72-dim pose vector

    smpl_out = {
        "betas": smpl_item["betas"],
        "thetas": thetas, 
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": tra,  # Use loaded tra as transl (aligned with new cam_trans)
        "rot": rot,
        "tra": tra,
        "name":image_path,
    }

    

    cam_info = CVCameraInfo(
        rt=extr,
        intrinsics=intrinsics,
        uid=idx, 
        bg=bg, 
        image=image, 
        image_path=image_path, 
        image_name=image_path.stem, 
        width=crop_width, 
        height=crop_height, 
        timestep=idx, 
        camera_id=idx,
        mask=crop_mask,
    )

    return cam_info, smpl_out

def loadSMPLXItem(idx, smplx_path, image_path):
    smplx_item = dict(np.load(smplx_path))

    # we are loading cropped images
    with Image.open(image_path) as img_file:
        image = img_file.copy()

    crop_width, crop_height = image.size

    #print('CROP',crop_width, crop_height)

    bg = np.array([1, 1, 1])

    orig_resolution = np.array([crop_height, crop_width])

    crop_box = None

    # Load intrinsics directly from NPZ, extracting scalars
    fx = smplx_item["fx"][0][0]
    fy = smplx_item["fy"][0][0]
    cx = smplx_item["cx"][0][0]
    cy = smplx_item["cy"][0][0]

    # if '02_025_0130' in str(image_path):
    #     print('Source Params:', fx, fy, cx, cy )
    #     print('smplx_path:', smplx_path)
    #     print('image_path:', image_path)


    scale_x = crop_width / 896.0
    scale_y = crop_height / 896.0

    # scale_x = 1.0
    # scale_y = 1.0

    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y

    #print('Scaled Source Params:', fx, fy, cx, cy )
    # if '02_025_0130' in str(image_path):
    #     print('Scaled Source Params:', fx, fy, cx, cy )

    # print("IMAGE", image.size)
    # print("fx, fy, cx, cy:", fx, fy, cx, cy)

    # if the image is cropped, get outcropping mask (unchanged)
    crop_mask = np.ones((crop_height, crop_width), dtype=bool)

    # Reconstruct extr from 'rot' (global_orient as axis-angle) and 'tra' (transl)
    global_orient = smplx_item["rot"]  # axis-angle vector (3,)
    tra = smplx_item["tra"].squeeze()  # translation vector (3,)
    
    #tra[1] = -tra[1]

    # Convert axis-angle to rotation matrix using batch_rodrigues
    # from flowface.flame.utils import batch_rodrigues
    # rot = batch_rodrigues(torch.from_numpy(global_orient)[None]).numpy()[0]  # 3x3 rotation matrix

    extr = np.eye(4)
    #extr[:3, :3] = rot  # Set rotation part
    rot = np.eye(3)
    #extr[:3, 3] = tra   # Set translation part

    # OPENCV_TO_OPENGL_CAMERA_CONVENTION = np.array([[1, 0, 0, 0],
    #                                            [0, -1, 0, 0],
    #                                            [0, 0, -1, 0],
    #                                            [0, 0, 0, 1]])

    # extr = OPENCV_TO_OPENGL_CAMERA_CONVENTION @ extr
    # extr = np.linalg.inv(extr)

    
    #tra = [-0.00661328, -0.3282768, 1.9570272]
    # tra[0] =-0.00661328
    # tra[1] = 0.3854
    # tra[2] = 1.9570272

    #extr[:3, 3] = tra 
    #extr = np.eye(4)
    #smpl_output.vertices += smpl_item["transl"]  # Add transl to model verts

    intrinsics = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
    )

    body_pose = smplx_item["body_pose"].reshape(-1)  # Flatten (21, 3) to (63,)
    jaw_pose = smplx_item["jaw_pose"]            # (3,)
    leye_pose = np.zeros(3)                      # Set to zeros since missing
    reye_pose = np.zeros(3)                      # Set to zeros since missing
    left_hand_pose = smplx_item["left_hand_pose"].reshape(-1)  # Flatten (15, 3) to (45,)
    right_hand_pose = smplx_item["right_hand_pose"].reshape(-1)  # Flatten (15, 3) to (45,)
    expression = smplx_item["expression"]        # (10,)
    betas = smplx_item["betas"]                  # (10,)


    #print(image_path, betas[:10])
    # Full pose vector: concatenate all pose parameters (total 165 dims, flattened)
    thetas = np.concatenate([global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose])

    smplx_out = {
        "betas": betas,
        "expression": expression,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": tra,  # Use loaded tra as transl
        "thetas": thetas, 
        "rot": rot,
        "tra": tra,
        "name": image_path,
    }

    


    cam_info = CVCameraInfo(
        rt=extr,
        intrinsics=intrinsics,
        uid=idx, 
        bg=bg, 
        image=image, 
        image_path=image_path, 
        image_name=image_path.stem, 
        width=crop_width, 
        height=crop_height, 
        timestep=idx, 
        camera_id=idx,
        mask=crop_mask,
    )
    
    # if '02_025_0130' in str(image_path):
    #     print('02_025_0130', smplx_out, cam_info)


    return cam_info, smplx_out





def loadSMPLItemOld(idx, smpl_path, image_path):
    smpl_item = dict(np.load(smpl_path))

    # we are loading cropped images
    with Image.open(image_path) as img_file:
        image = img_file.copy()

    crop_width, crop_height = image.size

    bg = np.array([1, 1, 1])

    orig_resolution = np.array([crop_height, crop_width])

    crop_box = None

    #print("orig_resolution", orig_resolution)

    focal_length = np.sqrt(crop_width**2 + crop_height**2)  
    fx, fy = focal_length, focal_length
    cx, cy = crop_width / 2.0, crop_height / 2.0



    # if the image is cropped, get outcropping mask
    crop_mask = np.ones((crop_height, crop_width), dtype=bool)

    cam = smpl_item["cam"]
    #print("SMPL cam:", cam)
    tra = convert_cam_to_3d_trans(cam)
    tra[1]  -= 0.20
    tra[0] += 0.035 
    #tra[1] = 0.0  
    #tra[2] = -tra[2]
    # s, tx, ty = cam[0], cam[1], cam[2]
    # depth = 5.0  
    # tra = np.array([tx, ty, depth])  
    rot = np.eye(3)  

    #from scipy.spatial.transform import Rotation as R
    #flip_rot = R.from_euler('x', 180, degrees=True).as_matrix()  # [[1,0,0],[0,-1,0],[0,0,-1]]
    #rot = flip_rot @ np.eye(3)  

    extr = np.eye(4)  
    extr[:3, :3] = rot  # Set rotation part
    extr[:3, 3] = tra   # Set translation part
    #extr[:3,:3] =[[1,0,0],[0,1,0],[0,0,-1]] #@ extr[:3,:3]  # Flip the z-axis
    #extr[:3, 2] = [0, 0, -1]

    intrinsics = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]],
    )


    global_orient = smpl_item["global_orient"]
    body_pose = smpl_item["body_pose"]
    thetas = np.concatenate([global_orient, body_pose])  # (72,) full pose
    smpl_out = {
        "betas": smpl_item["betas"],
        "thetas": thetas, 
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": np.zeros(3),  
        "rot": rot,
        "tra": tra,
    }

    cam_info = CVCameraInfo(
        rt=extr,
        intrinsics=intrinsics,
        uid=idx, 
        bg=bg, 
        image=image, 
        image_path=image_path, 
        image_name=image_path.stem, 
        width=crop_width, 
        height=crop_height, 
        timestep=idx, 
        camera_id=idx,
        mask=crop_mask,
    )

    return cam_info, smpl_out




def readCAP4DImageSet(path: Path, cam_id_offset=0):
    flame_paths = sorted(list((path / "flame").glob("*.npz")))
    img_paths = sorted(list((path / "images").glob("*.*")))
    
    cameras = []
    meshes = []
    
    #print(len(flame_paths),len(img_paths))
    #print("FLAME_PATH", flame_paths)
    #print("IMG", img_paths)
    print("FLAME PATHS:", len(flame_paths))
    print("IMG PATHS:", len(img_paths))
    assert len(flame_paths) > 0 and len(img_paths) == len(flame_paths)

    for frame_id in tqdm(range(len(flame_paths))):        
        camera, mesh = loadCAP4DItem(
            frame_id + cam_id_offset, 
            flame_paths[frame_id], 
            img_paths[frame_id], 
        )
        cameras.append(camera)
        meshes.append(mesh)

    return cameras, meshes


def readSMPLImageSet(path: Path, cam_id_offset=0):
    smpl_paths = sorted(list((path / "smpl").glob("*.npz")))
    img_paths = sorted(list((path / "images").glob("*.*")))
    
    cameras = []
    meshes = []
    
    #print(len(smpl_paths),len(img_paths))
    #print("SMPL_PATH", smpl_paths)
    #print("IMG", img_paths)
    # print("readSMPLImageSet FLAME PATHS:", len(smpl_paths))
    # print("IMG PATHS:", len(img_paths))
    assert len(smpl_paths) > 0 and len(img_paths) == len(smpl_paths)

    for frame_id in tqdm(range(len(smpl_paths))):        
        camera, mesh = loadSMPLItem(
            frame_id + cam_id_offset, 
            smpl_paths[frame_id], 
            img_paths[frame_id], 
        )
        # print("Loaded camera:", camera.uid, camera.image_name)
        # print("Camera intrinsics:", camera.intrinsics)
        # print("Camera rt:", camera.rt)
        # print("Camera", camera)
        # print("Mesh body pose :", mesh)
        #print("Mesh global orient :", mesh)

        cameras.append(camera)
        meshes.append(mesh)
    
    # for camera in cameras:
    #     print('Camera name:', camera.image_name)
    # for mesh in meshes:
    #     print('Mesh name:', mesh['name'])

    return cameras, meshes



def readSMPLXImageSet(path: Path, cam_id_offset=0):
    smplx_paths = sorted(list((path / "smplx").glob("*.npz")))
    img_paths = sorted(list((path / "images").glob("*.*")))
    
    #print(smplx_paths)
    #print(img_paths)


    cameras = []
    meshes = []
    
    #print(len(smpl_paths),len(img_paths))
    #print("SMPL_PATH", smpl_paths)
    #print("IMG", img_paths)
    # print("SMPLX PATHS:", len(smplx_paths))
    # print("IMG PATHS:", len(img_paths))
    assert len(smplx_paths) > 0 and len(img_paths) == len(smplx_paths)

    for frame_id in tqdm(range(len(smplx_paths))):  
        #print('Loading SMPLX frame:', frame_id)
        camera, mesh = loadSMPLXItem(
            frame_id + cam_id_offset, 
            smplx_paths[frame_id], 
            img_paths[frame_id], 
        )


        # print("Loaded camera:", camera.uid, camera.image_name)
        # print("Camera intrinsics:", camera.intrinsics)
        # print("Camera rt:", camera.rt)
        # print("Camera", camera)
        # print("Mesh body pose :", mesh)
        #print("Mesh global orient :", mesh)

        smplx_id = smplx_paths[frame_id].stem.split("_img")[0]   
        img_id   = img_paths[frame_id].stem.split("_img")[0]

        # if smplx_id != img_id:
        #     print("!!! MISMATCH !!!", smplx_id, img_id)

        cameras.append(camera)
        meshes.append(mesh)


    # for camera in cameras:
    #     if '02_025_0130' in str(camera.image_name):
    #         print('Camera:', camera)
    # for mesh in meshes:
    #     #if '02_025_0130' in str(mesh['name']):
    #     #    print('Mesh:', mesh) 
    #     print('Mesh name:', mesh['name'], mesh['betas'], mesh['global_orient'], frame_id)   

    # print('Total SMPLX cameras loaded:', len(cameras) )
    # print('Total SMPLX meshes loaded:', len(meshes) )

    return cameras, meshes


def readSMPLDrivingSequence(paths: Dict[str, Any], cam_id_offset=0):
    fit_path = paths["animation_path"]

    print(f"Loading target sequence from {fit_path}")
    
    fit = dict(np.load(paths["animation_path"]))

    n_frames = fit["body_pose"].shape[0]

    if "cam_trajectory_path" in paths and paths["cam_trajectory_path"] is not None:
        cam_traj_path = paths["cam_trajectory_path"]
        print(f"Loading camera trajectory from {cam_traj_path}")
        cam_trajectory = dict(np.load(cam_traj_path))

        extr_list = cam_trajectory["extr"]
        fx_list = cam_trajectory["fx"]
        fy_list = cam_trajectory["fy"]
        cx_list = cam_trajectory["cx"]
        cy_list = cam_trajectory["cy"]
        assert extr_list.shape[0] >= n_frames, "number of frames in the camera trajectory must be greater or equal to the driving sequence"

        resolution = cam_trajectory["resolution"]
    else:
        # If no camera trajectory, define a default fixed camera extrinsics (e.g., identity with z-translation for view)
        # Assuming static camera; adjust default extr if needed
        default_extr = np.eye(4)
        default_extr[2, 3] = -2.0  # Move camera back along z for better view (adjust as needed)
        extr_list = np.repeat(default_extr[None], n_frames, axis=0)
        
        fx_list = fit["fx"][[0]].repeat(n_frames, axis=0)
        fy_list = fit["fy"][[0]].repeat(n_frames, axis=0)
        cx_list = fit["cx"][[0]].repeat(n_frames, axis=0)
        cy_list = fit["cy"][[0]].repeat(n_frames, axis=0)

        resolution = fit["resolution"]

    cameras = []
    meshes = []

    for frame_id in tqdm(range(n_frames)):
        # Use global_orient as rot (axis-angle) and T as tra for reverse_transform
        rot = fit["global_orient"][frame_id]  # Axis-angle
        tra = fit["T"][frame_id]

        extr, rot, tra = reverse_transform(
            extr_list[frame_id],
            rot,
            tra,
        )

        intrinsics = np.array(
            [[fx_list[frame_id, 0], 0, cx_list[frame_id, 0]],
             [0, fy_list[frame_id, 0], cy_list[frame_id, 0]],
             [0, 0, 1]],
        )

        global_orient = fit["global_orient"][frame_id]
        body_pose = fit["body_pose"][frame_id]
        thetas = np.concatenate([global_orient, body_pose])

        smpl_out = {
            "betas": fit["betas"],  # Shared across frames or per-frame if array
            "thetas": thetas,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "transl": tra,  # Adjusted tra
            "rot": rot,  # Adjusted rot (zeroed in reverse_transform)
            "tra": tra,  # Adjusted tra (zeroed)
        }

        cam_info = CVCameraInfo(
            rt=extr,
            intrinsics=intrinsics,
            uid=cam_id_offset + frame_id,
            bg=None,
            image=None,
            image_path=None,
            image_name=None,
            width=resolution[1],
            height=resolution[0],
            timestep=cam_id_offset + frame_id,
            camera_id=cam_id_offset + frame_id,
            mask=None,
        )

        meshes.append(smpl_out)
        cameras.append(cam_info)

    return cameras, meshes

def readSMPLXDrivingSequence(paths: Dict[str, Any], cam_id_offset=0):
    fit_path = paths["animation_path"]

    print(f"Loading target sequence from {fit_path}")
    
    fit = dict(np.load(paths["animation_path"]))

    # Determine number of frames from one of the per-frame arrays (e.g., body_pose)
    n_frames = fit["body_pose"].shape[0]

    if "cam_trajectory_path" in paths and paths["cam_trajectory_path"] is not None:
        cam_traj_path = paths["cam_trajectory_path"]
        print(f"Loading camera trajectory from {cam_traj_path}")
        cam_trajectory = dict(np.load(cam_traj_path))

        extr_list = cam_trajectory["extr"]
        fx_list = cam_trajectory["fx"]
        fy_list = cam_trajectory["fy"]
        cx_list = cam_trajectory["cx"]
        cy_list = cam_trajectory["cy"]
        assert extr_list.shape[0] >= n_frames, "number of frames in the camera trajectory must be greater or equal to the driving sequence"

        resolution = cam_trajectory["resolution"]
    else:
        # If no camera trajectory, define a default fixed camera extrinsics (e.g., identity with z-translation for view)
        # Assuming static camera; adjust default extr if needed
        default_extr = np.eye(4)
        default_extr[2, 3] = -2.0  # Move camera back along z for better view (adjust as needed)
        extr_list = np.repeat(default_extr[None], n_frames, axis=0)
        
        fx_list = fit["fx"][[0]].repeat(n_frames, axis=0)
        fy_list = fit["fy"][[0]].repeat(n_frames, axis=0)
        cx_list = fit["cx"][[0]].repeat(n_frames, axis=0)
        cy_list = fit["cy"][[0]].repeat(n_frames, axis=0)

        resolution = fit["resolution"]

    cameras = []
    meshes = []

    for frame_id in tqdm(range(n_frames)):
        # Use global_orient as rot (axis-angle) and tra as tra for reverse_transform
        rot = fit["global_orient"][frame_id]  # Axis-angle
        tra = fit["tra"][frame_id].squeeze()  # Translation vector

        extr, rot, tra = reverse_transform(
            extr_list[frame_id],
            rot,
            tra,
        )

        # Adjust intrinsics based on resolution scaling (assuming original intrinsics are for 896 resolution)
        scale_x = resolution[1] / 896.0  # width
        scale_y = resolution[0] / 896.0  # height
        fx = fx_list[frame_id, 0] * scale_x
        fy = fy_list[frame_id, 0] * scale_y
        cx = cx_list[frame_id, 0] * scale_x
        cy = cy_list[frame_id, 0] * scale_y

        intrinsics = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]],
        )

        # Extract per-frame SMPL-X parameters
        global_orient = fit["global_orient"][frame_id]
        body_pose = fit["body_pose"][frame_id].reshape(-1)  # Flatten if needed
        jaw_pose = fit["jaw_pose"][frame_id]
        leye_pose = fit["leye_pose"][frame_id] if "leye_pose" in fit else np.zeros(3)
        reye_pose = fit["reye_pose"][frame_id] if "reye_pose" in fit else np.zeros(3)
        left_hand_pose = fit["left_hand_pose"][frame_id].reshape(-1)
        right_hand_pose = fit["right_hand_pose"][frame_id].reshape(-1)
        expression = fit["expression"][frame_id] if "expression" in fit else np.zeros(10)
        betas = fit["betas"][frame_id] if fit["betas"].ndim > 1 else fit["betas"]  # Shared or per-frame

        # Full pose vector: concatenate all pose parameters
        thetas = np.concatenate([global_orient, body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose])

        smplx_out = {
            "betas": betas,
            "expression": expression,
            "jaw_pose": jaw_pose,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "body_pose": body_pose,
            "global_orient": global_orient,
            "transl": tra,  # Adjusted tra
            "thetas": thetas, 
            "rot": rot,  # Adjusted rot (zeroed in reverse_transform)
            "tra": tra,  # Adjusted tra (zeroed)
        }

        cam_info = CVCameraInfo(
            rt=extr,
            intrinsics=intrinsics,
            uid=cam_id_offset + frame_id,
            bg=None,
            image=None,
            image_path=None,
            image_name=None,
            width=resolution[1],
            height=resolution[0],
            timestep=cam_id_offset + frame_id,
            camera_id=cam_id_offset + frame_id,
            mask=None,
        )

        meshes.append(smplx_out)
        cameras.append(cam_info)

    return cameras, meshes


def readCAP4DDrivingSequence(paths: Dict[str, Any], cam_id_offset=0):
    fit_path = paths["animation_path"]

    print(f"Loading target sequence from {fit_path}")
    
    fit = dict(np.load(paths["animation_path"]))

    n_frames = fit["expr"].shape[0]

    if "cam_trajectory_path" in paths and paths["cam_trajectory_path"] is not None:
        cam_traj_path = paths["cam_trajectory_path"]
        print(f"Loading camera trajectory from {cam_traj_path}")
        cam_trajectory = dict(np.load(cam_traj_path))

        extr_list = cam_trajectory["extr"]
        fx_list = cam_trajectory["fx"]
        fy_list = cam_trajectory["fy"]
        cx_list = cam_trajectory["cx"]
        cy_list = cam_trajectory["cy"]
        assert extr_list.shape[0] >= n_frames, "number of frames in the"
        " camera trajectory must be greater or equal to the driving sequence"

        resolution = cam_trajectory["resolution"]
    else:
        # select first camera of driving sequence and repeat (static camera)
        extr_list = fit["extr"][[0]].repeat(n_frames, axis=0)  
        fx_list = fit["fx"][[0]].repeat(n_frames, axis=0)
        fy_list = fit["fy"][[0]].repeat(n_frames, axis=0)
        cx_list = fit["cx"][[0]].repeat(n_frames, axis=0)
        cy_list = fit["cy"][[0]].repeat(n_frames, axis=0)

        resolution = fit["resolutions"][0]

    cameras = []
    meshes = []

    for frame_id in tqdm(range(n_frames)):
        extr, rot, tra = reverse_transform(
            extr_list[frame_id],
            fit["rot"][frame_id],
            fit["tra"][frame_id],
        )

        intrinsics = np.array(
            [[fx_list[frame_id, 0], 0, cx_list[frame_id, 0]],
            [0, fy_list[frame_id, 0], cy_list[frame_id, 0]],
            [0, 0, 1]],
        )

        flame_out = {
            "shape": np.zeros(150),  # shape is set to zero since we don't need it anyways!
            "expr": fit["expr"][frame_id], 
            "eye_rot": fit["eye_rot"][frame_id],
            "rot": rot,
            "tra": tra,
        }

        cam_info = CVCameraInfo(
            rt=extr,
            intrinsics=intrinsics,
            uid=cam_id_offset+frame_id, 
            bg=None, 
            image=None, 
            image_path=None, 
            image_name=None, 
            width=resolution[1], 
            height=resolution[0], 
            timestep=cam_id_offset+frame_id, 
            camera_id=cam_id_offset+frame_id,
            mask=None,
        )

        meshes.append(flame_out)
        cameras.append(cam_info)

    return cameras, meshes

def loadSMPLDataset(
    source_paths, 
    target_paths: Optional[Dict[str, str]] = None, 
    val_ratio=0.1,
    n_max_val_images=10,
):
    cameras = []
    meshes = []
    if source_paths is not None:
        for source_path in source_paths:
            source_path = Path(source_path)
            assert source_path.exists(), f"Source path does not exist: {source_path}"
            print(f"Loading dataset from {source_path}")
            cameras_, meshes_ = readSMPLImageSet(source_path, cam_id_offset=len(cameras))  
            cameras += cameras_
            meshes += meshes_

    n_frames = len(cameras)
    n_val = max(1, min(n_max_val_images, int(n_frames * val_ratio)))

    train_cameras = cameras[:-n_val]
    train_meshes = meshes
    val_cameras = cameras[-n_val:]

    print("Number of validation cameras:", len(val_cameras))
    print("Number of train cameras:", len(train_cameras))

    test_cameras = val_cameras
    #val_cameras = cameras[:n_val]
    test_meshes = []

    tgt_meshes = []
    tgt_cameras = []

    # now forbiden the animation
    # if target_paths is not None:
    #     tgt_cameras, tgt_meshes = readSMPLDrivingSequence(  
    #         target_paths, 
    #         cam_id_offset=len(train_meshes)+len(test_meshes)
    #     )
    # print("TARGET PATHS:", target_paths)
    if target_paths is not None:
        tgt_cameras, tgt_meshes = readSMPLDrivingSequence(
            target_paths, 
            cam_id_offset=len(train_meshes)+len(test_meshes)
        )



    
    scene_info = SceneInfo(
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        val_cameras=val_cameras,
        train_meshes=train_meshes,
        test_meshes=test_meshes,
        nerf_normalization={"radius": 2.},
        ply_path=None,
        tgt_meshes=tgt_meshes,
        tgt_cameras=tgt_cameras,
    )

    return scene_info



def loadSMPLXDataset(
    source_paths, 
    target_paths: Optional[Dict[str, str]] = None, 
    val_ratio=0.1,
    n_max_val_images=10,
):
    cameras = []
    meshes = []
    if source_paths is not None:
        for source_path in source_paths:
            source_path = Path(source_path)
            assert source_path.exists(), f"Source path does not exist: {source_path}"
            print(f"SMPLX Loading dataset from {source_path}")
            cameras_, meshes_ = readSMPLXImageSet(source_path, cam_id_offset=len(cameras))  
            cameras += cameras_
            meshes += meshes_

    n_frames = len(cameras)
    n_val = max(1, min(n_max_val_images, int(n_frames * val_ratio)))

    train_cameras = cameras[:-n_val]
    train_meshes = meshes
    val_cameras = cameras[-n_val:]

    # for debug
    #train_cameras = val_cameras

    print("Number of validation cameras:", len(val_cameras))
    print("Number of train cameras:", len(train_cameras))

    test_cameras = val_cameras
    #val_cameras = cameras[:n_val]
    test_meshes = []

    tgt_meshes = []
    tgt_cameras = []

    # now forbiden the animation
    # if target_paths is not None:
    #     tgt_cameras, tgt_meshes = readSMPLDrivingSequence(  
    #         target_paths, 
    #         cam_id_offset=len(train_meshes)+len(test_meshes)
    #     )
    # print("TARGET PATHS:", target_paths)
    if target_paths is not None:
        tgt_cameras, tgt_meshes = readSMPLXDrivingSequence(
            target_paths, 
            cam_id_offset=len(train_meshes)+len(test_meshes)
        )

    print("Loaded target SMPLX sequence with", len(tgt_cameras), "cameras and", len(tgt_meshes), "meshes")

    
    scene_info = SceneInfo(
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        val_cameras=val_cameras,
        train_meshes=train_meshes,
        test_meshes=test_meshes,
        nerf_normalization={"radius": 2.},
        ply_path=None,
        tgt_meshes=tgt_meshes,
        tgt_cameras=tgt_cameras,
    )

    return scene_info


def loadCAP4DDataset(
    source_paths, 
    target_paths: Optional[Dict[str, str]] = None, 
    val_ratio=0.1,
    n_max_val_images=10,
):
    cameras = []
    meshes = []
    if source_paths is not None:
        for source_path in source_paths:
            source_path = Path(source_path)

            assert source_path.exists(), f"Source path does not exist: {source_path}"

            print(f"Loading dataset from {source_path}")
            cameras_, meshes_ = readCAP4DImageSet(source_path, cam_id_offset=len(cameras))

            cameras += cameras_
            meshes += meshes_

    n_frames = len(cameras)
    n_val = max(1, min(n_max_val_images, int(n_frames * val_ratio)))

    train_cameras = cameras[:-n_val]  # select the last n cameras as validation cams
    train_meshes = meshes
    val_cameras = cameras[-n_val:]

    print("Number of validation cameras:", len(val_cameras))
    print("Number of train cameras:", len(train_cameras))

    test_cameras = val_cameras
    val_cameras = cameras[:n_val]  # These are training cameras
    test_meshes = []

    tgt_meshes = []
    tgt_cameras = []
    if target_paths is not None:
        tgt_cameras, tgt_meshes = readCAP4DDrivingSequence(
            target_paths, 
            cam_id_offset=len(train_meshes)+len(test_meshes)
        )
    
    scene_info = SceneInfo(
        train_cameras=train_cameras,
        test_cameras=test_cameras,
        val_cameras=val_cameras,
        train_meshes=train_meshes,
        test_meshes=test_meshes,
        nerf_normalization={"radius": 1.},
        ply_path=None,
        tgt_meshes=tgt_meshes,
        tgt_cameras=tgt_cameras,
    )

    ## ...
    return scene_info
