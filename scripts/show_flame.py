import numpy as np
import torch
import open3d as o3d
import pickle
import sys

# -----------------------------
# FLAME 顶点生成类（简化版）
# -----------------------------
class SimpleFLAME:
    def __init__(self, flame_pkl_path):
        with open(flame_pkl_path, 'rb') as f:
            flame_data = pickle.load(f, encoding='latin1')
        self.v_template = torch.tensor(flame_data['v_template']).float()   # [V,3]
        self.shapedirs   = torch.tensor(flame_data['shapedirs']).float()    # [V,3,N_shape]
        self.exprdirs    = torch.tensor(flame_data['exprdirs']).float()     # [V,3,N_expr]
        self.faces       = torch.tensor(flame_data['f'].astype(np.int32))   # [F,3]

    def __call__(self, shape_params, expr_params):
        """
        shape_params: [N_shape]
        expr_params: [N_t, N_expr]
        output: verts [N_t, V, 3]
        """
        shape_offsets = torch.einsum('vn,n->v3', self.shapedirs, shape_params)
        template = self.v_template + shape_offsets  # [V,3]
        verts_list = []
        for expr in expr_params:
            expr_offsets = torch.einsum('vn,n->v3', self.exprdirs, expr)
            verts = template + expr_offsets
            verts_list.append(verts)
        verts = torch.stack(verts_list, dim=0)  # [N_t, V, 3]
        return verts, self.faces

# -----------------------------
# 可视化函数
# -----------------------------
def visualize(vertices, faces=None, frame_idx=0):
    verts = vertices[frame_idx].numpy()
    if faces is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces.numpy())
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        o3d.visualization.draw_geometries([pcd])

# -----------------------------
# 主函数
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python show_flame_win.py <fit.npz> <flame.pkl> [frame_idx]")
        sys.exit(0)

    npz_path = sys.argv[1]
    pkl_path = sys.argv[2]
    frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # load fit.npz
    fit = np.load(npz_path, allow_pickle=True)
    shape = torch.tensor(fit['shape']).float()       # [N_shape]
    expr  = torch.tensor(fit['expr']).float()        # [N_t, N_expr]

    # 构造 FLAME
    flame = SimpleFLAME(pkl_path)

    # 生成顶点
    verts, faces = flame(shape, expr)
    print("Generated verts:", verts.shape, "faces:", faces.shape)

    # 可视化
    visualize(verts, faces, frame_idx)
