import torch
import einops


def get_pos_enc(
    n_dim,
    resolution,
):
    coords = torch.stack(torch.meshgrid(torch.arange(resolution), torch.arange(resolution), indexing='ij'), dim=0)
    coords = coords / resolution * 2. - 1.

    assert n_dim % 2 == 0
    n_ch = n_dim // 2
    freqs = 2. ** torch.linspace(0., n_ch - 1, steps=n_ch)

    pos_xyz = coords[..., None] * freqs[None, None, None]

    pos_emb = torch.cat([torch.sin(pos_xyz), torch.cos(pos_xyz)], dim=-1)
    pos_emb = einops.rearrange(pos_emb, "c h w f -> (c f) h w")

    return pos_emb