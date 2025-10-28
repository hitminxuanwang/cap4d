
import torch
import numpy as np
import pickle as pkl

from cap4d.flame.flame import CAP4DFlameSkinner, compute_flame

# Paths
npz_path = 'fit.npz' # Your fit.npz file
flame_pkl_path = '/home/mathen/mathen/Cap4d_Avatar/data/assets/flame/flame2023_no_jaw.pkl' # Full path to FLAME pkl file
blink_blendshape_path = '/home/mathen/mathen/Cap4d_Avatar/data/assets/flame/blink_blendshape.npy'  # Full path to blink_blendshape.npy

# Load fit_3d from npz
fit_3d = dict(np.load(npz_path))
# Ensure arrays are numpy and convert only numeric ones to float32
for k in fit_3d:
    if isinstance(fit_3d[k], np.ndarray) and np.issubdtype(fit_3d[k].dtype, np.number):
        fit_3d[k] = fit_3d[k].astype(np.float32) if fit_3d[k].ndim > 0 else fit_3d[k]

# Initialize the flame model
flame = CAP4DFlameSkinner(
    flame_pkl_path=flame_pkl_path,
    n_shape_params=fit_3d["shape"].shape[-1],
    n_expr_params=fit_3d["expr"].shape[-1],
    blink_blendshape_path=blink_blendshape_path,
    add_mouth=False, # Adjust based on your model
    add_lower_jaw=False # Since no_jaw variant
)

# Compute flame outputs
flame_results = compute_flame(flame, fit_3d)
verts = flame_results['verts_3d'][0] # [V, 3] for single frame

# Get faces from pkl
with open(flame_pkl_path, 'rb') as f:
    flame_data = pkl.load(f, encoding='latin1')
faces = flame_data['f'] # FLAME faces from pkl

# Export to OBJ (geometry only)
obj_path = 'flame_visual.obj'
with open(obj_path, 'w') as f:
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print(f"OBJ file saved to {obj_path}")