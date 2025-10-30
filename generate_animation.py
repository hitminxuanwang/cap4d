import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation

# Parameters
n_frames = 100  # Number of frames in the animation
output_npz = 'right_hand_wave_animation.npz'  # Output file name
smpl_model_path = 'data/assets/smpl/SMPL_NEUTRAL.pkl'  # Path to SMPL model (adjust if needed)
resolution = np.array([1080, 1080])  # Image resolution (for camera intrinsics, can adjust)

# Load SMPL model
smpl_model = smplx.SMPL(model_path=smpl_model_path, batch_size=n_frames)

# Initialize parameters
betas = torch.zeros(n_frames, 10)  # Shape parameters (neutral)

global_orient = torch.zeros(n_frames, 3)  # Root orientation (neutral)

body_pose = torch.zeros(n_frames, 69)  # Body pose (23 joints x 3)

# Animate the right arm for a waving motion:
# - Raise the right shoulder slightly to lift the arm (joint 17: indices 48:51)
# - Bend the right elbow a bit (joint 19: indices 54:57)
# - Wave the right wrist side-to-side (joint 21: indices 60:63)

# Base pose: Raise arm to waving position (static for simplicity)
# Right shoulder: Rotate to lift arm sideways (e.g., around y-axis)
body_pose[:, 48:51] = torch.tensor([[0.0, 0.0, np.pi / 3]]).repeat(n_frames, 1)  # Lift right shoulder (adjust angles as needed)

# Right elbow: Bend slightly
body_pose[:, 54:57] = torch.tensor([[0.0, 0.0, np.pi / 4]]).repeat(n_frames, 1)  # Bend elbow

# Waving motion: Oscillate right wrist around z-axis (side-to-side wave)
for i in range(n_frames):
    t = i / n_frames * 4 * np.pi  # Two full waves over the animation
    wave_angle = np.sin(t) * np.pi / 6  # Small oscillation +/- 30 degrees
    body_pose[i, 60:63] = torch.tensor([wave_angle, 0.0, 0.0])  # Rotate wrist around x-axis for up-down, or z for side-to-side (adjust axis)

# Translation: Move back for better view
transl = torch.zeros(n_frames, 3)
transl[:, 2] = 2.0  # Z-translation

# Compute SMPL output (vertices, etc., but we only need params for NPZ)
with torch.no_grad():
    smpl_output = smpl_model(betas=betas, global_orient=global_orient, body_pose=body_pose, transl=transl)

# Prepare rotation matrices from global_orient (axis-angle to matrix)
R = np.array([Rotation.from_rotvec(go).as_matrix() for go in global_orient.numpy()])

# Camera intrinsics (simple pinhole, adjust if needed)
fx = np.full((n_frames, 1), resolution[1] * 0.5)
fy = np.full((n_frames, 1), resolution[0] * 0.5)
cx = np.full((n_frames, 1), resolution[1] / 2)
cy = np.full((n_frames, 1), resolution[0] / 2)

# Save to NPZ (matching the format in your original script)
np.savez(output_npz,
         betas=betas.numpy()[0],  # Betas are shared across frames
         global_orient=global_orient.numpy(),
         body_pose=body_pose.numpy(),
         T=transl.numpy(),
         R=R,
         fx=fx, fy=fy, cx=cx, cy=cy,
         resolution=resolution)

print(f"Right hand waving animation NPZ saved to: {output_npz}")