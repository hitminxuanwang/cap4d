import einops
import torch
import numpy as np
from functools import partial
from einops import rearrange, repeat
from torchvision.utils import make_grid

from controlnet.ldm.models.diffusion.ddpm import LatentDiffusion
from controlnet.ldm.util import exists, default
from controlnet.ldm.modules.diffusionmodules.util import make_beta_schedule
from controlnet.ldm.models.diffusion.ddim import DDIMSampler
from cap4d.mmdm.utils import shift_schedule, enforce_zero_terminal_snr


class MMLDM(LatentDiffusion):
    """
    Class for morphable multi-view latent diffusion model
    """

    def __init__(
        self, 
        control_key, 
        only_mid_control, 
        n_frames, 
        *args, 
        cfg_probability=0.1,
        shift_schedule=False,
        sqrt_shift=False,
        zero_snr_shift=True,
        minus_one_shift=True,
        negative_shift=False,
        **kwargs
    ):
        self.n_frames = n_frames
        self.shift_schedule = shift_schedule
        self.sqrt_shift = sqrt_shift
        self.minus_one_shift = minus_one_shift
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.cfg_probability = cfg_probability
        self.negative_shift = negative_shift
        self.zero_snr_shift = zero_snr_shift

        super().__init__(*args, **kwargs)

    # @torch.no_grad()
    def get_input(self, batch, k, bs=None, force_conditional=False, *args, **kwargs):
        with torch.no_grad():
            # From DDPM
            x = batch[k]
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b t h w c -> b t c h w')
            x = x.to(memory_format=torch.contiguous_format) # .float()  CONTIGUOUS
            
            # From LatentDiffusion
            if bs is not None:
                x = x[:bs]
            b_, t_ = x.shape[:2]
            x_flat = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            encoder_posterior = self.encode_first_stage(x_flat)
            z_flat = self.get_first_stage_encoding(encoder_posterior).detach()
            z = einops.rearrange(z_flat, '(b t) c h w -> b t c h w', b=b_)

            # Add gt z to control
            batch[self.control_key]['z'] = z.detach()
            
            c_uncond = self.get_unconditional_conditioning(batch[self.control_key])

            if "mask" in batch:
                loss_mask = batch["mask"]
            else:
                loss_mask = None

        c_cond = self.get_learned_conditioning(batch[self.control_key])

        if not force_conditional:
            is_uncond = torch.rand(b_, device=x.device) < self.cfg_probability  # do a mix with probability
            is_cond = torch.logical_not(is_uncond)
            control = {}
            for key in c_cond:
                control[key] = (
                    einops.einsum(c_uncond[key], is_uncond, 'b ..., b -> b ...') + 
                    einops.einsum(c_cond[key], is_cond, 'b ..., b -> b ...')
                )
        else:
            control = c_cond

        # New stuff
        assert isinstance(control, dict)

        if bs is not None:
            for key in control:
                control[key] = control[key][:bs]
        
        return z, dict(c_concat=[control], c_uncond=[c_uncond], mask=loss_mask)
    
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False):
        b_, t_ = z.shape[:2]
        z = einops.rearrange(z, 'b t c h w -> (b t) c h w')
        z = super().decode_first_stage(z, predict_cids)
        return einops.rearrange(z, '(b t) c h w -> b t c h w', b=b_)
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, x.shape[:2], device=self.device).long()

        assert c is not None
        assert not self.shorten_cond_schedule

        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = None  # remove text conditioning

        assert len(cond['c_concat']) == 1

        control = cond['c_concat'][0]
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps
    
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        b_, t_ = t.shape[:2]
        t_flat = einops.rearrange(t, 'b t -> (b t)')
        noise_flat = einops.rearrange(noise, 'b t c h w -> (b t) c h w')
        x_start_flat = einops.rearrange(x_start, 'b t c h w -> (b t) c h w')
        x_noisy_flat = self.q_sample(x_start=x_start_flat, t=t_flat, noise=noise_flat)
        x_noisy = einops.rearrange(x_noisy_flat, '(b t) c h w -> b t c h w', b=b_)
        
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        assert self.parameterization == 'eps'
        target = noise

        loss_simple = self.get_loss(model_output, target, mean=False)
        # Mask loss by references
        loss_simple = loss_simple.mean(dim=[2, 3, 4])

        ref_mask = torch.logical_not(cond['c_concat'][0]['ref_mask'])
        loss_simple_mean = (loss_simple * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1)
        
        # Losses updated with time dimension
        loss_dict.update({f'{prefix}/loss_simple': loss_simple_mean.mean()})

        logvar_t = self.logvar[t]  # .to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        loss = (loss * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1)
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1)
        loss_vlb = loss_vlb.mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    # @torch.no_grad()
    def get_learned_conditioning(self, c):
        return self.cond_stage_model(c, unconditional=False)

    @torch.no_grad()
    def get_unconditional_conditioning(self, c):
        return self.cond_stage_model(c, unconditional=True)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N, force_conditional=True)
        c_cat = c["c_concat"][0]
        c_uncond = c["c_uncond"][0]
        for key in c_cat:
            c_cat[key] = c_cat[key][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t], device=self.device), '1 -> b', b=n_row)
                    t = t.long()  # .to(self.device)
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [None]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=c_uncond,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        # b, c, h, w = cond["c_concat"][0].shape
        # shape = (self.channels, h // 8, w // 8)
        shape = (self.n_frames, 4, self.image_size, self.image_size) # 64, 64)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list([param for param in self.cond_stage_model.parameters() if param.requires_grad])

            for n, param in self.cond_stage_model.named_parameters():
                if param.requires_grad:
                    print(n)
        
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            # self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            # self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
            
        if self.zero_snr_shift:
            print(f"Enforcing zero terminal SNR in noise schedule.")
            betas = enforce_zero_terminal_snr(betas)

        betas[betas > 0.99] = 0.99

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        if self.shift_schedule:
            n_gen = self.n_frames
            if self.minus_one_shift:
                n_gen = n_gen - 1  # we are generating only n_total - 1 frames technically
            shift_ratio = (64 ** 2) / (self.image_size ** 2 * n_gen)
            if self.negative_shift:
                shift_ratio = 1. / shift_ratio
            if self.sqrt_shift:
                shift_ratio = np.sqrt(shift_ratio)
            new_alpha_cumprod, new_betas = shift_schedule(alphas_cumprod, shift_ratio=shift_ratio)

            print(f"Shifted log psnr of noise schedule by {shift_ratio}.")

            alphas = 1. - new_betas
            betas = new_betas
            alphas_cumprod = new_alpha_cumprod

        print("Using non persistent schedule buffers.")

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas), persistent=False)
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod), persistent=False)
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev), persistent=False)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)), persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)), persistent=False)
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)), persistent=False)
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)), persistent=False)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)), persistent=False)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance), persistent=False)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))), persistent=False)
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)), persistent=False)
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)), persistent=False)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

        # From LatentDiffusion
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()
