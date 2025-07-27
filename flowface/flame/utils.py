from typing import Dict

import torch
import einops


# define the transform matrix between opencv and pytorch3d coordinate conventions
OPENCV2PYTORCH3D = torch.eye(4)
OPENCV2PYTORCH3D[1, 1] = -1
OPENCV2PYTORCH3D[2, 2] = -1


def dot(
    x: torch.Tensor,
    y: torch.Tensor,
    dim=-1,
    keepdim=False,
) -> torch.Tensor:
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


def safe_length(
    x: torch.Tensor,
    dim=-1,
    keepdim=False,
    eps=1e-20,
) -> torch.Tensor:
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x, dim=dim, keepdim=keepdim), min=eps))


def transform_vertices(
    transform: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """
    Transforms a list of vertices by a transform.

    Parameters
    ----------
    transform: torch.Tensor [B, 4, 4]
    vertices: torch.Tensor [B, N, 3]

    Returns
    -------
    transformed_vertices: torch.Tensor [B, N, 3]
    """
    transformed_verts = transform[:, :3, :3] @ vertices.permute(0, 2, 1)
    transformed_verts = transformed_verts + transform[:, :3, [3]]
    return transformed_verts.permute(0, 2, 1)


def batch_rodrigues(
    rot_vecs: torch.Tensor, epsilon=1e-8  # (B, 3)
) -> torch.Tensor:  # (B, 3, 3)
    """Calculates the rotation matrices for a batch of rotation vectors.
    Reference:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    Parameters
    ----------
    rot_vecs: torch.tensor Bx3
        array of B axis-angle vectors
    Returns
    -------
    R: torch.tensor Bx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = safe_length(rot_vecs, keepdim=True, eps=epsilon)  # (B, 1)
    rot_dir = rot_vecs / angle  # (B, 3)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)  # each (B, 1)

    zeros = torch.zeros(
        (batch_size, 1), dtype=torch.float32, device=device
    )  # (B, 1)
    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1
    )  # (B, 9)
    K = K.view(batch_size, 3, 3)  # (B, 3, 3)

    ident: torch.Tensor = torch.eye(
        3, dtype=torch.float32, device=device
    ).unsqueeze(
        dim=0
    )  # (1, 3, 3)
    cos = torch.unsqueeze(torch.cos(angle), dim=1)  # (B, 1, 1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)  # (B, 1, 1)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def project_vertices(
    verts_3d: torch.Tensor, 
    cam_parameters: Dict,
) -> torch.Tensor:
    """
    Project 3D vertices onto screen given a number of camera parameters
    verts_3d (N_t, V, 3)
    cam_parameters (dictionary):
        with N_c as number of cameras
        fx (N_c)
        fy (N_c)
        cx (N_c)
        cy (N_c)
        extr (N_c, 4, 4)

    output: verts_2d (N_c, N_t, V, 3)
    """
    # perform transform for each camera
    extr = cam_parameters["extr"]
    verts_3d_cam = einops.einsum(extr[:, :3, :3], verts_3d, "N_c i j, N_t V j -> N_c N_t V i")
    verts_3d_cam = verts_3d_cam + extr[:, None, None, :3, 3]  # N_c N_t V 3

    fx = cam_parameters["fx"][:, None]  # N_c 1 1
    fy = cam_parameters["fy"][:, None]
    cx = cam_parameters["cx"][:, None]
    cy = cam_parameters["cy"][:, None]

    verts_2d = torch.stack(
        [
            verts_3d_cam[..., 0] / verts_3d_cam[..., 2] * fx + cx,
            verts_3d_cam[..., 1] / verts_3d_cam[..., 2] * fy + cy,
            verts_3d_cam[..., 2] / verts_3d_cam[..., 2].mean(dim=-1)[..., None] * (fx + fy) / 2,
        ],
        dim=-1,
    )

    return verts_2d
