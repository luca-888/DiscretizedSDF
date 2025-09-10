#
# 最小修改方案的训练脚本
# 利用现有的 CUDA 代码，只修改 Python 层的参数处理
#

import os
import sys
import uuid
import torch
import torchvision
from time import time
from tqdm import tqdm
from random import randint
from gaussian_renderer import RENDER_DICT, render_lighting
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import apply_depth_colormap, psnr
from utils.loss_utils import (l1_loss, predicted_normal_loss, ssim,
                               bilateral_smooth_loss, base_smooth_loss)
from fused_ssim import fused_ssim 
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from render import render_set
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, load_iteration):
    render_fn = RENDER_DICT[pipe.gaussian_type]
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    
    # 🆕 添加 disable_pbr 参数传递
    gaussians = GaussianModel(dataset.sh_degree, dataset.env_mode, dataset.env_res,
                              dataset.use_sdf, opt.metallic, opt.sphere_init,
                              disable_pbr=getattr(dataset, 'disable_pbr', False))
    
    if load_iteration > -1:
        scene = Scene(dataset, gaussians, load_iteration=load_iteration)
    else:
        scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, scene)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(load_iteration+1, opt.iterations + 1), desc="Training progress")
    progress_bar_range = opt.iterations + 1
    start_time = time()
    
    if getattr(gaussians, 'disable_pbr', False):
        print("🚫 Using simplified rendering (PBR disabled)")

    # Training loop
    for iteration in range(load_iteration+1, opt.iterations + 1):
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == opt.densify_from_iter:
            background = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_fn(viewpoint_cam, scene, pipe, background, debug=False)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        mask = torch.ones_like(gt_image[0:1, :, :])
        if viewpoint_cam.alpha is not None:
            mask = viewpoint_cam.alpha.cuda()
        
        # 基础损失计算
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), mask.unsqueeze(0)))
        losses_extra = {}

        # 🆕 条件性 PBR 损失计算
        if not getattr(gaussians, 'disable_pbr', False):
            # 原有的 PBR 损失计算
            if opt.lambda_brdf_smoothness > 0:
                l_roughness = bilateral_smooth_loss(render_pkg['roughness'], gt_image, mask)
                l_albedo = bilateral_smooth_loss(render_pkg['albedo'], gt_image, mask)
                l_metallic = bilateral_smooth_loss(render_pkg['metallic'], gt_image, mask)
                losses_extra['brdf_smoothness'] = l_roughness + l_albedo + l_metallic
        else:
            if iteration % 1000 == 0:
                print("🚫 Skipping PBR losses (disabled)")

        # 几何相关损失 (保持不变)
        depth_expected = render_pkg['depth']
        depth_med, depth_var = render_pkg['depth'], render_pkg['depth']
        alpha_value, normal_image = render_pkg['alpha'], render_pkg['normal']
        
        # SDF-related losses
        if gaussians.use_sdf:
            l_sdf_dev = torch.abs(gaussians.get_sdf).mean()
            loss += opt.lambda_dev * l_sdf_dev
            losses_extra['sdf_deviation'] = l_sdf_dev

        # Normal losses
        if opt.lambda_predicted_normal > 0:
            l_normal_ref = predicted_normal_loss(render_pkg['normal_ref'])
            l_normal_rend = predicted_normal_loss(render_pkg['normal'])
            loss += opt.lambda_predicted_normal * (l_normal_ref + l_normal_rend)
            losses_extra['predicted_normal'] = l_normal_ref + l_normal_rend

        # Zero-one losses
        if opt.lambda_zero_one > 0:
            if 'zero_one' in render_pkg:
                l_zero_one = (render_pkg['zero_one']).mean()
                loss += opt.lambda_zero_one * l_zero_one
                losses_extra['zero_one'] = l_zero_one

        # Distortion losses (simplified version skips this)
        if opt.lambda_distortion > 0 and not getattr(gaussians, 'disable_pbr', False):
            l_distortion = (render_pkg['distortion']).mean()
            loss += opt.lambda_distortion * l_distortion
            losses_extra['distortion'] = l_distortion

        for k, v in losses_extra.items():
            loss += v

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start=start_time,  testing_iterations=testing_iterations, 
                          scene=scene, render_fn=render_fn, pipe=pipe, dataset=dataset, background=background)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in testing_iterations):
                dataset.model_path = os.path.join(dataset.model_path, 'evaluate')
                dataset.model_path = dataset.model_path.replace('/evaluate/evaluate', '/evaluate')
                validation_configs = ({'name':'test', 'cameras' : scene.getTestCameras()}, 
                                    {'name':'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0.0
                        psnr_test = 0.0
                        for idx, viewpoint in enumerate(config['cameras']):
                            render_results = render_fn(viewpoint, scene, pipe, background, debug=False)
                            image = torch.clamp(render_results["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            if tb_writer and (idx < 5):
                                tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                                if iteration == testing_iterations[0]:
                                    tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            l1_test += l1_loss(image, gt_image).mean().double()
                            psnr_test += psnr(image, gt_image).mean().double()
                        psnr_test /= len(config['cameras'])
                        l1_test /= len(config['cameras'])          
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                        if tb_writer:
                            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                            tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def prepare_output_and_logger(dataset, opt, pipe):
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start, testing_iterations, scene, render_fn, pipe, dataset, background):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', time() - iter_start, iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # 🆕 添加 PBR 禁用开关
    parser.add_argument("--disable_pbr", action="store_true", help="Disable PBR calculations for simplified training")
    args = parser.parse_args(sys.argv[1:])
    
    # 🆕 设置 disable_pbr 到 dataset
    args.disable_pbr = args.disable_pbr
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")