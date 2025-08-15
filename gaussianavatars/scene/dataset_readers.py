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


def readCAP4DImageSet(path: Path, cam_id_offset=0):
    flame_paths = sorted(list((path / "flame").glob("*.npz")))
    img_paths = sorted(list((path / "images").glob("*.*")))
    
    cameras = []
    meshes = []
    
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
