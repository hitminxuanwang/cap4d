import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation

# 参数设置
folder_path = './npzs'  # 替换为你的NPZ文件所在文件夹路径
output_npz = 'combined_animation.npz'  # 输出动画NPZ文件名
resolution = np.array([1080, 1920])  # 相机分辨率（可调整）

# 获取文件夹中所有NPZ文件，并按文件名排序（假设文件名如frame_001.npz，按数字顺序）
npz_files = sorted(glob.glob(os.path.join(folder_path, '*.npz')))

n_frames = len(npz_files)
if n_frames == 0:
    raise ValueError("文件夹中没有找到NPZ文件")

# 初始化参数数组
betas = None  # betas通常共享，取第一个文件的值
global_orient = np.zeros((n_frames, 3))
body_pose = np.zeros((n_frames, 69))
transl = np.zeros((n_frames, 3))  # T作为transl
R = np.zeros((n_frames, 3, 3))  # 旋转矩阵

# 逐个读取NPZ文件并提取参数
for i, npz_file in enumerate(npz_files):
    data = np.load(npz_file)
    
    if betas is None:
        betas = data['betas']  # 取第一个文件的betas（假设所有帧共享）
    
    global_orient[i] = data['global_orient'].flatten()  # 确保扁平化
    body_pose[i] = data['body_pose'].flatten()  # 确保扁平化
    transl[i] = data['T'].flatten()  # 扁平化 (3,1) 到 (3,)
    R[i] = data['R']  # 如果NPZ中有R，直接取；否则可从global_orient计算

# 如果NPZ中没有R，可以从global_orient计算（注释掉如果已有）
# R = np.array([Rotation.from_rotvec(go).as_matrix() for go in global_orient])

# 相机内参（简单针孔模型，可调整）
fx = np.full((n_frames, 1), resolution[1] * 0.5)
fy = np.full((n_frames, 1), resolution[0] * 0.5)
cx = np.full((n_frames, 1), resolution[1] / 2)
cy = np.full((n_frames, 1), resolution[0] / 2)

# 保存为单个NPZ文件
np.savez(output_npz,
         betas=betas,  # 共享的betas
         global_orient=global_orient,
         body_pose=body_pose,
         T=transl,
         R=R,
         fx=fx, fy=fy, cx=cx, cy=cy,
         resolution=resolution)

print(f"Combined animation NPZ saved to: {output_npz}")