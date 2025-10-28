import torch
from gaussianavatars.scene.cap4d_gaussian_model import SMPLGaussianModel
from gaussianavatars.scene.scene import SMPLScene
from gaussianavatars.gaussian_renderer.gsplat_renderer import render
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

# Define paths (replace with actual values)
model_path = Path("examples/output/full_body/avatar/")  # Your model directory
pth_path = model_path / "chkpnt100000.pth"  # Specific checkpoint file, e.g., chkpnt10000.pth
source_paths = ["examples/output/full_body/"]  # List of source directories for meshes/images

# Load config for model_params
avatar_config = OmegaConf.load(model_path / "config_dump.yaml")
model_params = avatar_config["model_params"]

# Load checkpoint
checkpoint = torch.load(pth_path, weights_only=False)
model_weights, loaded_iter = checkpoint  # Assume tuple structure (weights, iter)

# Initialize model and scene
smpl_gaussians = SMPLGaussianModel(model_params)
scene = SMPLScene(model_path=str(model_path), source_paths=source_paths, gaussians=smpl_gaussians)
smpl_gaussians.restore(model_weights)  # Restore weights

# Select camera and timestep (adjust as needed)
camera = scene.getTestCameras()[0] if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()[0]
smpl_gaussians.select_mesh_by_timestep(camera.timestep)

# Background
background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

# Render
smpl_gaussians.eval()
render_out = render(camera, smpl_gaussians, background)
image = render_out["render"].clamp(0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy() * 255.0

# Save
Image.fromarray(image.astype(np.uint8)).save("rendered_image.png")