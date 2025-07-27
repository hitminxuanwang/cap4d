import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from cap4d.mmdm.conditioning.mesh2img import PropRenderer


class PositionalEncoding(nn.Module):
    def __init__(self, channels_per_dim):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()

        assert channels_per_dim % 2 == 0
        n_ch = channels_per_dim // 2
        freqs = 2. ** torch.linspace(0., n_ch - 1, steps=n_ch)

        self.register_buffer("freqs", freqs)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (b, h, w, [x, y, z]), x, y, z should be [0, 1]
        :return: Positional Encoding Matrix of size (b, h, w, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        pos_xyz = tensor[..., None] * self.freqs[None, None, None, None]

        pos_emb = torch.cat([torch.sin(pos_xyz), torch.cos(pos_xyz)], dim=-1)
        pos_emb = einops.rearrange(pos_emb, "b h w c f -> b h w (c f)")

        return pos_emb


class CAP4DConditioning(nn.Module):
    def __init__(
        self,
        image_size=64,
        positional_channels=42,
        positional_multiplier=1.,
        super_resolution=2,
        use_ray_directions=True,
        use_expr_deformation=True,
        use_crop_mask=False,
        std_expr_deformation=0.0104,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        assert super_resolution >=1 and super_resolution % 1 == 0
        self.super_resolution = super_resolution
        self.positional_channels = positional_channels
        self.positional_multiplier = positional_multiplier
        self.use_ray_directions = use_ray_directions
        self.use_expr_deformation = use_expr_deformation
        self.std_expr_deformation = std_expr_deformation
        self.use_crop_mask = use_crop_mask

        assert positional_channels % 3 == 0
        self.pos_encoding = PositionalEncoding(positional_channels // 3)
        self.renderer = PropRenderer()

    def forward(self, batch, unconditional=True):
        verts = batch["verts_2d"]
        offsets = batch["offsets_3d"]
        ref_mask = batch["reference_mask"][:, :, None]
        B, T = verts.shape[:2]

        z_input = None
        if "z" in batch:
            z_input = batch["z"]

        img_size = self.image_size

        if unconditional:
            total_channels = self.positional_channels + 1 # positional and ref mask
            if self.use_crop_mask:
                total_channels += 1
            if self.use_ray_directions:
                total_channels += 3
            if self.use_expr_deformation: 
                total_channels += 3
            pose_pos_enc = torch.zeros((B, T, img_size, img_size, total_channels), device=verts.device)
            if z_input is not None:
                z_input = z_input * 0.
        else:
            with torch.no_grad():
                verts = einops.rearrange(verts, 'b t n v -> (b t) n v')
                offsets = einops.rearrange(offsets, 'b t n v -> (b t) n v')
                offsets = offsets / self.std_expr_deformation  # normalize offset magnitude

                pose_map, mask = self.renderer.render(
                    verts, 
                    (img_size * self.super_resolution, img_size * self.super_resolution),
                    prop=offsets if self.use_expr_deformation else None,
                )

                if self.use_expr_deformation:
                    # extract last three channels which are offsets
                    pose_map, offsets = pose_map.split([3, 3], dim=-1)

                # Need to unnormalize from [-1, 1] to [0, resolution]
                # pos_enc = self.pos_encoding((uv_img + 1.) / 2. * self.image_size * self.positional_multiplier)

                pose_pos_enc = self.pos_encoding(pose_map * self.positional_multiplier)

                if self.use_expr_deformation:
                    # append expression offset
                    pose_pos_enc = torch.cat([pose_pos_enc, offsets], dim=-1)

                pose_pos_enc = pose_pos_enc * mask  # mask values not rendered

                # downscale pos_enc if we use super resolution
                pose_pos_enc = einops.rearrange(pose_pos_enc, 'bt h w c -> bt c h w')
                pose_pos_enc = F.interpolate(pose_pos_enc, (img_size, img_size), mode="area")
                pose_pos_enc = einops.rearrange(pose_pos_enc, '(b t) c h w -> b t h w c', b=B)

                if self.use_ray_directions:
                    # concat ray map
                    ray_map = batch["ray_map"]
                    ray_map = einops.rearrange(ray_map, 'b t c h w -> b t h w c')
                    pose_pos_enc = torch.cat([pose_pos_enc, ray_map], dim=-1)

                # concat ref mask
                ref_mask_reshape = einops.rearrange(ref_mask, 'b t c h w -> b t h w c')
                pose_pos_enc = torch.cat([pose_pos_enc, ref_mask_reshape], dim=-1)

                if self.use_crop_mask:
                    crop_mask = batch["out_crop_mask"][..., None]
                    pose_pos_enc = torch.cat([pose_pos_enc, crop_mask], dim=-1)

        return {
            "pos_enc": pose_pos_enc,
            "z_input": z_input,
            "ref_mask": ref_mask,
        }

    def get_vis(self, enc):
        visualizations = {}

        n_pos = self.positional_channels // 3

        counter = 0

        pos_enc = enc[..., 0:self.positional_channels]

        # for i in [n_pos-1]:
        for i in range(n_pos-2, n_pos):
            visualizations[f"pose_map_{i}"] = pos_enc[..., [i, i + n_pos, i + n_pos * 2]]

        counter = self.positional_channels

        if self.use_expr_deformation:
            visualizations["expr_disp"] = enc[..., counter:counter+3]
            counter += 3

        if self.use_ray_directions:
            visualizations["ray_map"] = enc[..., counter:counter+3]
            counter += 3

        visualizations["ref_mask"] = enc[..., [counter] * 3]
        counter += 1

        if self.use_crop_mask:
            visualizations["crop_mask"] = enc[..., [counter] * 3]
            counter += 1

        return visualizations
