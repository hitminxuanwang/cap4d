import torch
import torch.nn as nn
import einops

from controlnet.ldm.modules.diffusionmodules.openaimodel import UNetModel
from controlnet.ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

from cap4d.mmdm.net.attention import SpatioTemporalTransformer


class MMDMUnetModel(UNetModel):
    def __init__(
        self, 
        *args, 
        time_steps,
        condition_channels=50, 
        model_channels=320,
        image_size=32,
        context_dim=1024,
        temporal_mode="3d",  # ["3d", "temporal"]
        **kwargs,
    ):
        assert temporal_mode in ["3d", "temporal"]
        self.temporal_mode = temporal_mode
        self.time_steps = time_steps
        self.use_context = False

        super().__init__(*args, model_channels=model_channels, image_size=image_size, context_dim=context_dim, **kwargs)

        self.cond_linear = zero_module(nn.Linear(condition_channels, model_channels))

    def create_attention_block(
        self, 
        ch,
        mult,
        use_checkpoint,
        num_heads,
        dim_head,
        transformer_depth,
        context_dim,
        disable_self_attn,
        use_linear,
        use_new_attention_order,
        use_spatial_transformer,
    ):
        if self.temporal_mode == "temporal":
            temporal_connection_type = "temporal"
        elif self.temporal_mode == "3d":
            if mult >= 2:
                temporal_connection_type = "3d"
            else:
                temporal_connection_type = "none"

        return SpatioTemporalTransformer(
            ch, 
            num_heads, 
            dim_head, 
            use_context=self.use_context,
            context_dim=context_dim,
            temporal_connection_type=temporal_connection_type,
            num_timesteps=self.time_steps,
        )

    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        """
        x (b t h w c): input latent
        control (b t h w c_cond): input conditioning signal
        """
        
        z_input = control["z_input"]

        ref_mask = control["ref_mask"]
        # ground truth noise output
        x_input = x - z_input  

        ref_mask_inv = torch.logical_not(ref_mask)

        # replace with input latents where available
        x = z_input * ref_mask + x * ref_mask_inv

        # Disabling context cross attention for MMDM
        assert context == None

        # Flatten time dimension for processing
        b_, t_ = x.shape[:2]
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        timesteps = einops.rearrange(timesteps, 'b t -> (b t)')

        pos_enc = einops.rearrange(control["pos_enc"], 'b t h w c -> (b t) h w c')
        pos_enc = pos_enc.type(self.dtype)
        pos_embedding = self.cond_linear(pos_enc)
        pos_embedding = pos_embedding.permute(0, 3, 1, 2)

        hs = []
        
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.type(self.dtype)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)

            if pos_embedding is not None:
                h += pos_embedding
                pos_embedding = None

            hs.append(h)

        h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = self.out(h)
        h = h.type(x.dtype)

        # Unflatten time dimension
        h = einops.rearrange(h, '(b t) c h w -> b t c h w', b=b_)
        
        # replace with input latents where available
        h = x_input * ref_mask + h * ref_mask_inv

        return h