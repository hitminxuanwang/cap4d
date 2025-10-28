from pathlib import Path
import copy
import shutil

import torch
from torch.utils.data import DataLoader
import gc
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from cap4d.mmdm.sampler import StochasticIOSampler
from cap4d.inference.data.reference_data import ReferenceDataset
from cap4d.inference.data.generation_data import GenerationDataset
from cap4d.inference.utils import (
    get_condition_from_dataloader,
    load_model,
    save_visualization,
    save_flame_params,
    convert_and_save_latent_images,
)

# Hardcoded paths and defaults
REF_DATA_PATH = Path("examples/input/felix/")
CONFIG_PATH = Path("configs/generation/multi_ref.yaml")
OUTPUT_PATH = Path("examples/output/run1/")
DEVICE = "cuda"
VISUALIZE_CONDITIONING = True

@torch.no_grad()
def load_config_and_paths():
    gen_config = OmegaConf.load(CONFIG_PATH)
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    output_ref_path = OUTPUT_PATH / "reference_images"
    output_ref_path.mkdir(exist_ok=True)
    output_gen_path = OUTPUT_PATH / "generated_images"
    output_gen_path.mkdir(exist_ok=True)
    seed_everything(gen_config["seed"])
    shutil.copy(CONFIG_PATH, OUTPUT_PATH / "mmdm_config_dump.yaml")
    return gen_config, output_ref_path, output_gen_path

@torch.no_grad()
def create_dataloaders(gen_config):
    print("Creating dataloaders")
    refset = ReferenceDataset(REF_DATA_PATH, gen_config["resolution"])
    ref_dataloader = DataLoader(refset, num_workers=1, batch_size=1, shuffle=False)
    genset = GenerationDataset(
        generation_data_path=gen_config["generation_data"]["data_path"],
        reference_flame_item=refset.flame_list[0],
        n_samples=gen_config["generation_data"]["n_samples"],
        resolution=gen_config["resolution"],
        yaw_range=gen_config["generation_data"]["yaw_range"],
        pitch_range=gen_config["generation_data"]["pitch_range"],
        expr_factor=gen_config["generation_data"]["expr_factor"],
    )
    gen_dataloader = DataLoader(genset, num_workers=1, batch_size=1, shuffle=False)
    return ref_dataloader, gen_dataloader, refset

@torch.no_grad()
def setup_models(gen_config):
    model = load_model(Path(gen_config["ckpt_path"]))
    device_model_map = {}
    first_rank_model = None
    first_rank_device = None
    if "cuda" in DEVICE:
        for cuda_id in range(torch.cuda.device_count()):
            dev_key = f"cuda:{cuda_id}"
            device_model_map[dev_key] = copy.deepcopy(model).to(dev_key)
            if first_rank_model is None:
                first_rank_model = device_model_map[dev_key]
                first_rank_device = dev_key
    else:
        device_model_map[DEVICE] = model.to(DEVICE)
        first_rank_model = device_model_map[DEVICE]
        first_rank_device = DEVICE
    print(f"Done loading model.")
    return device_model_map, first_rank_model, first_rank_device

@torch.no_grad()
def extract_conditions(first_rank_model, ref_dataloader, gen_dataloader, output_ref_path, output_gen_path, gen_config):
    print(f"Loading reference dataset from {REF_DATA_PATH}")
    ref_data = get_condition_from_dataloader(first_rank_model, ref_dataloader, DEVICE)
    print(f"Loading generation dataset from {gen_config['generation_data']['data_path']}")
    gen_data = get_condition_from_dataloader(first_rank_model, gen_dataloader, DEVICE)
    if VISUALIZE_CONDITIONING:
        print("Saving visualization of conditioning images")
        save_visualization(ref_data["cond_vis_frames"], output_ref_path)
        save_visualization(gen_data["cond_vis_frames"], output_gen_path)
    print("Saving flame parameters")
    save_flame_params(ref_data["flame_params"], output_ref_path)
    save_flame_params(gen_data["flame_params"], output_gen_path)
    gc.collect()
    for key in ref_data["cond_frames"]:
        ref_data["cond_frames"][key] = torch.cat(ref_data["cond_frames"][key], dim=0)
        ref_data["uncond_frames"][key] = torch.cat(ref_data["uncond_frames"][key], dim=0)
        gen_data["cond_frames"][key] = torch.cat(gen_data["cond_frames"][key], dim=0)
        gen_data["uncond_frames"][key] = torch.cat(gen_data["uncond_frames"][key], dim=0)
    return ref_data, gen_data

@torch.no_grad()
def run_sampling(device_model_map, ref_data, gen_data, gen_config):
    print(f"Generating images on {len(list(device_model_map))} devices with stochastic I/O.")
    stochastic_io_sampler = StochasticIOSampler(device_model_map)
    z_gen = stochastic_io_sampler.sample(
        S=gen_config["n_ddim_steps"],
        ref_cond=ref_data["cond_frames"],
        ref_uncond=ref_data["uncond_frames"],
        gen_cond=gen_data["cond_frames"],
        gen_uncond=gen_data["uncond_frames"],
        latent_shape=(4, gen_config["resolution"] // 8, gen_config["resolution"] // 8),
        V=gen_config["V"],
        R_max=gen_config["R_max"],
        cfg_scale=gen_config["cfg_scale"],
    )
    print("Done generating.")
    gc.collect()
    z_gen = z_gen.cpu()
    z_ref = ref_data["cond_frames"]["z_input"].cpu()
    return z_gen, z_ref

@torch.no_grad()
def save_outputs(z_gen, z_ref, first_rank_model, first_rank_device, output_ref_path, output_gen_path):
    print(f"Saving reference images to {output_ref_path}/images")
    convert_and_save_latent_images(z_ref, first_rank_model, first_rank_device, output_ref_path)
    print(f"Saving generated images to {output_gen_path}/images")
    convert_and_save_latent_images(z_gen, first_rank_model, first_rank_device, output_gen_path)

@torch.no_grad()
def main():
    gen_config, output_ref_path, output_gen_path = load_config_and_paths()
    ref_dataloader, gen_dataloader, refset = create_dataloaders(gen_config)
    device_model_map, first_rank_model, first_rank_device = setup_models(gen_config)
    print('...........')
    ref_data, gen_data = extract_conditions(first_rank_model, ref_dataloader, gen_dataloader, output_ref_path, output_gen_path, gen_config)
    z_gen, z_ref = run_sampling(device_model_map, ref_data, gen_data, gen_config)
    # save_outputs(z_gen, z_ref, first_rank_model, first_rank_device, output_ref_path, output_gen_path)

if __name__ == "__main__":
    main()