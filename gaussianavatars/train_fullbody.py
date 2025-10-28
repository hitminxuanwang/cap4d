#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from gaussianavatars.utils.system_utils import searchForMaxIteration
from gaussianavatars.gaussian_renderer.gsplat_renderer import render
from gaussianavatars.scene.cap4d_gaussian_model import CAP4DGaussianModel, SMPLGaussianModel
from gaussianavatars.scene.scene import Scene, SMPLScene
from gaussianavatars.utils.loss_utils import l1_loss, ssim
from gaussianavatars.utils.general_utils import safe_state
from gaussianavatars.utils.image_utils import psnr, error_map
from gaussianavatars.lpipsPyTorch import LPIPS
from gaussianavatars.utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from PIL import Image
import numpy as np
import os

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    source_paths,
    model_path,
    model_params, 
    opt_params, 
    testing_iterations, 
    checkpoint_iterations, 
    load_existing_checkpoint, 
):
    first_iter = 0
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    smpl_guassians = SMPLGaussianModel(model_params)
    scene = SMPLScene(
        model_path=model_path, 
        source_paths=source_paths, 
        gaussians=smpl_guassians,
        shuffle=True,
    )
    smpl_guassians.training_setup(opt_params)





    
    ## debug
    # camera = scene.getTrainCameras()[0]
    # print("Camera", camera.image_name)

    # # camera.rt[:3,:3] = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]]).cuda()

    # # print("CameraRT",camera.rt)
    # # print(camera.intrinsics)

    # # print("=== After select_mesh_by_timestep ===")
    # # print("Timestep:", camera.timestep)  
    # # print("Face center shape:", smpl_guassians.face_center.shape if smpl_guassians.face_center is not None else "None")
    # # print("Face scaling mean:", smpl_guassians.face_scaling.mean().item() if smpl_guassians.face_scaling is not None else "None")
    # # print("Face orientation quat mean:", smpl_guassians.face_orien_quat.mean().item() if smpl_guassians.face_orien_quat is not None else "None")
    # # print("Computed XYZ (after binding) range:", smpl_guassians.get_xyz.min().item(), smpl_guassians.get_xyz.max().item())  
    # # print("Deform output mean (if exists):", smpl_guassians.deform_output.mean().item() if hasattr(smpl_guassians, 'deform_output') else "No deform")


    # # print("Opacity:",smpl_guassians.get_opacity.mean().item() if hasattr(smpl_guassians, 'get_opacity') else "No opacity")
    # # print("Scaling", smpl_guassians.get_scaling.mean().item() if hasattr(smpl_guassians, 'get_scaling') else "No scaling")
    # smpl_guassians.select_mesh_by_timestep(0)
    # background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    # #print("Sampled Gaussian positions:", smpl_guassians.get_xyz[:10].cpu().tolist())

    # render_out = render(camera, smpl_guassians, background)

    # render_image = render_out["render"].clamp(0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy() * 255.0
    # gt_image = camera.original_image.permute(1, 2, 0).cpu().numpy() * 255.0 if camera.original_image is not None else np.zeros_like(render_image)
    # #gt_image = Image.open("/home/mathen/mathen/Cap4d_Avatar/examples/output/full_body/images/image_0000.png")
    # gt_np = np.array(gt_image)

    # # Blend render onto GT with alpha=0.5
    # gt_pil = Image.fromarray(gt_np.astype(np.uint8))
    # render_pil = Image.fromarray(render_image.astype(np.uint8))
    # combined_image = Image.blend(gt_pil, render_pil, alpha=0.5)
    
    # #combined_image.save(os.path.join(model_path, f"./debug/debug_blended_{iteration}.png"))

    # #image = render_out["render"].clamp(0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy()*255.0
    # combined_image.save(os.path.join(model_path, f"init_render.png"))
    
    # Pre-training rendering for all loaded images to compare with GT
    # debug_dir = os.path.join(model_path, "debug_pretrain")
    # os.makedirs(debug_dir, exist_ok=True)
    # background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    # train_cameras = scene.getTrainCameras()
    # for idx, camera in enumerate(train_cameras):
    #     if smpl_guassians.binding is not None:
    #         smpl_guassians.select_mesh_by_timestep(camera.timestep)
        
    #     render_out = render(camera, smpl_guassians, background)
    #     render_image = render_out["render"].clamp(0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy() * 255.0
    #     gt_image = camera.original_image.permute(1, 2, 0).cpu().numpy() * 255.0 if camera.original_image is not None else np.zeros_like(render_image)
        
    #     # Save render image
    #     render_pil = Image.fromarray(render_image.astype(np.uint8))
    #     render_pil.save(os.path.join(debug_dir, f"render_{idx:04d}.png"))
        
    #     # Save GT image
    #     gt_pil = Image.fromarray(gt_image.astype(np.uint8))
    #     gt_pil.save(os.path.join(debug_dir, f"gt_{idx:04d}.png"))
        
    #     # Blend render onto GT with alpha=0.5
    #     blended_image = Image.blend(gt_pil, render_pil, alpha=0.5)
    #     blended_image.save(os.path.join(debug_dir, f"blended_{idx:04d}.png"))

    # return None

    # if prompted and if it exists, load existing checkpoint
    if load_existing_checkpoint:
        loaded_iter, chkpt_path = searchForMaxIteration(model_path)
        if loaded_iter is None:
            print("WARNING: No valid checkpoint found in ", model_path)
        else:
            print("Loading trained model at iteration {}".format(loaded_iter))
            (model_weights, first_iter) = torch.load(chkpt_path, weights_only=False)
            smpl_guassians.restore(model_weights, opt_params)
    
    lpips = LPIPS('vgg').cuda()

    bg_color = [1, 1, 1]  # force white background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(
        scene.getTrainCameras(), 
        batch_size=None, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True
    )
    iter_camera_train = iter(loader_camera_train)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_params["iterations"]), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt_params["iterations"] + 1):    
        iter_start.record()

        smpl_guassians.train()

        smpl_guassians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % opt_params["sh_warmup_iterations"] == 0:
            smpl_guassians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        # Set timestep and run FLAME model
        if smpl_guassians.binding != None:
            smpl_guassians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render Gaussians
        render_pkg = render(
            viewpoint_cam, 
            smpl_guassians, 
            background, 
        )
    
        #return None

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # load gt image and mask
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.mask[..., None].cuda().permute(2, 0, 1)
        assert mask.shape[1] == image.shape[1] and mask.shape[2] == image.shape[2]
        image = image * mask
        gt_image = gt_image * mask


        
        render_image = render_pkg["render"].clamp(0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy() * 255.0
        #gt_image = viewpoint_cam.original_image.permute(1, 2, 0).cpu().numpy() * 255.0 if viewpoint_cam.original_image is not None else np.zeros_like(render_image)
        
        # print("CameraTrans:/t", viewpoint_cam.trans, viewpoint_cam.scale)
        # print('CameraRT:/t', viewpoint_cam.rt)
        # print('Intrinsics:/t', viewpoint_cam.intrinsics)

        #if(iteration % 100 == 0):
        # if True:
        #     # Blend render onto GT with alpha=0.5
        #     # gt_image_cpu = gt_image.permute(1, 2, 0).cpu().numpy() * 255.0
        #     # gt_pil = Image.fromarray(gt_image_cpu.astype(np.uint8))
        #     # render_pil = Image.fromarray(render_image.astype(np.uint8))
        #     # combined_image = Image.blend(gt_pil, render_pil, alpha=0.6)
            
        #     # combined_image.save(os.path.join(model_path, f"debug/debug_blended_{iteration}.png"))
        #         gt_image_cpu = (gt_image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        #         gt_pil = Image.fromarray(gt_image_cpu)
        #         gt_pil.save(os.path.join(model_path, f"debug/gt_{iteration}.png"))


        #         if isinstance(render_image, torch.Tensor):
        #             render_image_cpu = (render_image.permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
        #         else:
        #             render_image_cpu = render_image.astype(np.uint8)
        #         render_pil = Image.fromarray(render_image_cpu)
        #         render_pil.save(os.path.join(model_path, f"debug/render_{iteration}.png"))



        #print(model_path)


        # Loss computation
        losses = {}

        lambda_lpips = 0.

        if iteration > opt_params["lpips_linear_start"]:
            lambda_lpips = (iteration - opt_params["lpips_linear_start"]) / (opt_params["lpips_linear_end"] - opt_params["lpips_linear_start"]) * opt_params["lambda_lpips_end"]
            lambda_lpips = min(lambda_lpips, opt_params["lambda_lpips_end"])
            losses['lpips'] = opt_params["w_lpips"] * lambda_lpips * lpips(image, gt_image)
        else:
            losses['lpips'] = torch.tensor(0., device="cuda")

        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt_params["lambda_dssim"]) * (1.0 - lambda_lpips)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt_params["lambda_dssim"] * (1.0 - lambda_lpips)

        if opt_params["metric_xyz"]:
            losses['xyz'] = F.relu((smpl_guassians._xyz*smpl_guassians.face_scaling[smpl_guassians.binding])[visibility_filter] - opt_params["threshold_xyz"]).norm(dim=1).mean() * opt_params["lambda_xyz"]
        else:
            losses['xyz'] = F.relu(smpl_guassians._xyz[visibility_filter].norm(dim=1) - opt_params["threshold_xyz"]).mean() * opt_params["lambda_xyz"]

        if opt_params["lambda_scale"] != 0:
            if opt_params["metric_scale"]:
                losses['scale'] = F.relu(smpl_guassians.get_scaling[visibility_filter] - opt_params["threshold_scale"]).norm(dim=1).mean() * opt_params["lambda_scale"]
            else:
                losses['scale'] = F.relu(torch.exp(smpl_guassians._scaling[visibility_filter]) - opt_params["threshold_scale"]).norm(dim=1).mean() * opt_params["lambda_scale"]

        if opt_params["lambda_laplacian"] != 0:
            losses['lap'] = smpl_guassians.compute_laplacian_loss() * opt_params["lambda_laplacian"]

        if opt_params["lambda_relative_deform"] != 0:
            losses['deform'] = smpl_guassians.compute_relative_deformation_loss() * opt_params["lambda_relative_deform"]

        if opt_params["lambda_relative_rot"] != 0:
            losses['rot'] = smpl_guassians.compute_relative_rotation_loss() * opt_params["lambda_relative_rot"]

        # if opt_params["lambda_neck"] != 0:
        #     losses['neck'] = smpl_guassians.compute_neck_loss() * opt_params["lambda_neck"]
        

        #losses['dummy'] = (image - gt_image).abs().mean() * 0.0
        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            smpl_guassians.eval()

            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt_params["iterations"]:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, 
                iteration, 
                losses, 
                iter_start.elapsed_time(iter_end), 
                testing_iterations, 
                scene, 
                render, 
                background,
                lpips,
            )

            # Densification
            if iteration < opt_params["densify_until_iter"]:
                # Keep track of max radii in image-space for pruning
                smpl_guassians.max_radii2D[visibility_filter] = torch.max(smpl_guassians.max_radii2D[visibility_filter], radii[visibility_filter])
                smpl_guassians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt_params["densify_from_iter"] and iteration % opt_params["densification_interval"] == 0:
                    size_threshold = 20 if iteration > opt_params["opacity_reset_interval"] else None
                    smpl_guassians.densify_and_prune(opt_params["densify_grad_threshold"], 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt_params["opacity_reset_interval"] == 0 or (iteration == opt_params["densify_from_iter"]):
                    smpl_guassians.reset_opacity()

            # Optimizer step
            if iteration < opt_params["iterations"]:
                smpl_guassians.optimizer_step()

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((smpl_guassians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training_report(
    tb_writer, 
    iteration, 
    losses, 
    elapsed, 
    testing_iterations, 
    scene: Scene, 
    renderFunc, 
    background,
    lpips: LPIPS,
):
    if tb_writer and iteration % 10 == 0:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].detach().item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].detach().item(), iteration)
        if 'lpips' in losses:
            tb_writer.add_scalar('train_loss_patches/lpips', losses['lpips'].detach().item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].detach().item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].detach().item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].detach().item(), iteration)
        if 'lap' in losses:
            tb_writer.add_scalar('train_loss_patches/lap', losses['lap'].detach().item(), iteration)
        if 'deform' in losses:
            tb_writer.add_scalar('train_loss_patches/deform', losses['deform'].detach().item(), iteration)
        if 'rot' in losses:
            tb_writer.add_scalar('train_loss_patches/rot', losses['rot'].detach().item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].detach().item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].detach().item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, background)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras'])) // num_vis_img) == 0:
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        
                        # Visualize U-Net expression dependent deformation (manually normalized)
                        deform = scene.gaussians.deform_output / 0.0108 / 2. + 0.5
                        tb_writer.add_images(config['name'] + f"_{vis_ct}/deform", deform, global_step=iteration)

                        vis_ct += 1
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_paths', type=str, nargs="*", 
                        help="List of source directories containing images and flame parameters")
    parser.add_argument('--model_path', type=str, 
                        help="Path to directory where the gaussian avatar model is saved.")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=10_000, 
                        help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[],
                        help="Extra testing iterations (for visualization).")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--load_existing_checkpoint", type=int, default=None,
                        help="Whether to load existing (newest) checkpoint in model_path")
    parser.add_argument("--config_path", type=str, default = None)
    args = parser.parse_args()

    print("Loading config from", args.config_path)
    config = OmegaConf.load(args.config_path)
    opt_params = config["opt_params"]
    model_params = config["model_params"]

    if args.interval > opt_params["iterations"]:
        args.interval = opt_params["iterations"] // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, opt_params["iterations"]+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, opt_params["iterations"]+1, args.interval)))

    print("Optimizing " + args.model_path)
    model_path = Path(args.model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(args.config_path, model_path / "config_dump.yaml")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        args.source_paths,
        args.model_path,
        model_params,
        opt_params,
        args.test_iterations, 
        args.checkpoint_iterations, 
        args.load_existing_checkpoint, 
    )

    # All done
    print("\nTraining complete.")
