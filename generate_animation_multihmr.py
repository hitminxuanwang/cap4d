import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation

# Parameter settings
folder_path = './npzs'  # Replace with the path to your NPZ files folder
output_npz = 'combined_animation.npz'  # Output animation NPZ filename
resolution = np.array([1080, 1920])  # Camera resolution (height, width, adjustable)

# Get all NPZ files in the folder and sort them by filename (assuming names like frame_001.npz, sorted numerically)
npz_files = sorted(glob.glob(os.path.join(folder_path, '*.npz')))

n_frames = len(npz_files)
if n_frames == 0:
    raise ValueError("No NPZ files found in the folder")

# Initialize parameter arrays
betas = None  # betas usually shared, take from the first file
global_orient = np.zeros((n_frames, 3))
body_pose = np.zeros((n_frames, 63))  # SMPLX body_pose: 21 joints * 3 = 63
jaw_pose = np.zeros((n_frames, 3))
leye_pose = np.zeros((n_frames, 3))
reye_pose = np.zeros((n_frames, 3))
left_hand_pose = np.zeros((n_frames, 45))  # 15 joints * 3 = 45
right_hand_pose = np.zeros((n_frames, 45))  # 15 joints * 3 = 45
expression = np.zeros((n_frames, 10))  # expression: 10 dims
transl = np.zeros((n_frames, 3))  # tra as transl
R = np.zeros((n_frames, 3, 3))  # Rotation matrix

# Read each NPZ file and extract parameters
for i, npz_file in enumerate(npz_files):
    data = np.load(npz_file)
    
    if betas is None:
        betas = data['betas']  # Take betas from the first file (assuming shared across frames)
    
    global_orient[i] = data['global_orient'].flatten()  # Ensure flattened
    body_pose[i] = data['body_pose'].flatten()  # Ensure flattened
    jaw_pose[i] = data['jaw_pose'].flatten()
    leye_pose[i] = data['leye_pose'].flatten() if 'leye_pose' in data else np.zeros(3)
    reye_pose[i] = data['reye_pose'].flatten() if 'reye_pose' in data else np.zeros(3)
    left_hand_pose[i] = data['left_hand_pose'].flatten()
    right_hand_pose[i] = data['right_hand_pose'].flatten()
    expression[i] = data['expression'].flatten() if 'expression' in data else np.zeros(10)
    transl[i] = data['tra'].flatten() if 'tra' in data else data['T'].flatten()  # Support tra or T
    if 'R' in data:
        R[i] = data['R']
    else:
        # If no R, compute from global_orient
        R[i] = Rotation.from_rotvec(global_orient[i]).as_matrix()

# Camera intrinsics (simple pinhole model, adjustable)
fx = np.full((n_frames, 1), resolution[1] * 0.5)
fy = np.full((n_frames, 1), resolution[0] * 0.5)
cx = np.full((n_frames, 1), resolution[1] / 2)
cy = np.full((n_frames, 1), resolution[0] / 2)

# Save to a single NPZ file
np.savez(output_npz,
         betas=betas,  # Shared betas
         global_orient=global_orient,
         body_pose=body_pose,
         jaw_pose=jaw_pose,
         leye_pose=leye_pose,
         reye_pose=reye_pose,
         left_hand_pose=left_hand_pose,
         right_hand_pose=right_hand_pose,
         expression=expression,
         tra=transl,  # Use tra as key, consistent with loadSMPLXItem
         R=R,
         fx=fx, fy=fy, cx=cx, cy=cy,
         resolution=resolution)

print(f"Combined animation NPZ saved to: {output_npz}")