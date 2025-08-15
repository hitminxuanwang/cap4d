import torch
import einops


def gen_uv_mesh(
    uv_mask,
):
    faces = []
    for x in range(uv_mask.shape[2] - 1):
        for y in range(uv_mask.shape[3] - 1):
            p00 = x + y * uv_mask.shape[2]
            p10 = x + 1 + y * uv_mask.shape[2]
            p01 = x + (y + 1) * uv_mask.shape[2]
            p11 = x + 1 + (y + 1) * uv_mask.shape[2]
            faces.append([p00, p01, p11])
            faces.append([p00, p11, p10])
    
    faces = torch.tensor(faces, device=uv_mask.device)
    vert_mask = einops.rearrange(uv_mask, 'b m h w -> (b h w) m').bool()
    face_mask = vert_mask[faces].min(dim=-2)[0][:, 0]
    faces = faces[face_mask, :]
    
    return faces