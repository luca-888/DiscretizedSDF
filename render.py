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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import RENDER_DICT, render_lighting
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel
from utils.image_utils import apply_depth_colormap
from scene.NVDIFFREC.util import save_image_raw
from utils.camera_utils import interpolate_camera
import subprocess
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

def render_lightings(model_path, name, iteration, scene, sample_num):
    gaussians = scene.gaussians
    lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)    
    sampled_indicies = torch.arange(gaussians.get_xyz.shape[0], dtype=torch.long)[:sample_num]
    for sampled_index in tqdm(sampled_indicies, desc="Rendering lighting progress"):
        lighting = render_lighting(gaussians, sampled_index=sampled_index)
        torchvision.utils.save_image(lighting, os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".png"))
        save_image_raw(os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".hdr"), lighting.permute(1,2,0).detach().cpu().numpy())

def render_set(model_path, name, iteration, views, scene, pipeline, background, batch_size=1, clear_cache=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_masks_path, exist_ok=True)
    render_fn = RENDER_DICT[pipeline.gaussian_type]

    # 分批处理视角以节省显存
    total_views = len(views)
    for batch_start in tqdm(range(0, total_views, batch_size), desc="Rendering batches"):
        batch_end = min(batch_start + batch_size, total_views)
        batch_views = views[batch_start:batch_end]
        
        for idx_in_batch, view in enumerate(batch_views):
            idx = batch_start + idx_in_batch
            
            torch.cuda.synchronize()
            render_pkg = render_fn(view, scene, pipeline, background, debug=False)
          
            torch.cuda.synchronize()

            gt = view.original_image[0:3, :, :]
            gt_alpha_mask = view.gt_alpha_mask
            torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt_alpha_mask, os.path.join(gt_masks_path, '{0:05d}'.format(idx) + ".png"))
            
            for k in render_pkg.keys():
                if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
                    continue
                save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
                makedirs(save_path, exist_ok=True)
                if k == "alpha":
                    render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
                if k == "depth":
                    render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
                elif "normal" in k:
                    render_pkg[k] = 0.5 + (0.5*render_pkg[k])
                torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))
            
            # 清理render_pkg以释放显存
            del render_pkg
            if clear_cache:
                torch.cuda.empty_cache()
        
        # 批次结束后清理缓存
        if clear_cache:
            torch.cuda.empty_cache()

def create_video_from_images(image_dir, output_path, fps=30):
    """使用ffmpeg从图像序列创建视频"""
    try:
        cmd = [
            'ffmpeg', '-y',  # -y 覆盖输出文件
            '-framerate', str(fps),
            '-i', os.path.join(image_dir, '%05d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # 高质量
            output_path
        ]
        
        print(f"创建视频: {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"视频创建成功: {output_path}")
            return True
        else:
            print(f"视频创建失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("错误: 未找到ffmpeg，请确保已安装ffmpeg")
        return False
    except Exception as e:
        print(f"创建视频时出错: {str(e)}")
        return False

def save_render_result(render_data):
    """并行保存渲染结果的工作函数"""
    render_pkg, gt, gt_alpha_mask, idx, paths = render_data
    render_path, gts_path, gt_masks_path, model_path, name, iteration = paths
    
    # 保存主要结果
    torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    torchvision.utils.save_image(gt_alpha_mask, os.path.join(gt_masks_path, '{0:05d}'.format(idx) + ".png"))
    
    # 保存其他渲染结果
    for k in render_pkg.keys():
        if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
            continue
        save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
        makedirs(save_path, exist_ok=True)
        if k == "alpha":
            render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
        if k == "depth":
            render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
        elif "normal" in k:
            render_pkg[k] = 0.5 + (0.5*render_pkg[k])
        torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))

def render_set_parallel(model_path, name, iteration, views, scene, pipeline, background, batch_size=1, clear_cache=False, num_workers=4):
    """并行渲染函数"""
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gt_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gt_masks_path, exist_ok=True)
    render_fn = RENDER_DICT[pipeline.gaussian_type]
    
    # 创建保存任务队列
    save_queue = queue.Queue()
    paths = (render_path, gts_path, gt_masks_path, model_path, name, iteration)
    
    def save_worker():
        """保存工作线程"""
        while True:
            item = save_queue.get()
            if item is None:
                break
            save_render_result((item[0], item[1], item[2], item[3], paths))
            save_queue.task_done()
    
    # 启动保存线程
    save_threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=save_worker)
        t.start()
        save_threads.append(t)
    
    try:
        # 分批处理视角
        total_views = len(views)
        for batch_start in tqdm(range(0, total_views, batch_size), desc="Rendering batches"):
            batch_end = min(batch_start + batch_size, total_views)
            batch_views = views[batch_start:batch_end]
            
            # 批量渲染（GPU部分保持串行）
            for idx_in_batch, view in enumerate(batch_views):
                idx = batch_start + idx_in_batch
                
                torch.cuda.synchronize()
                render_pkg = render_fn(view, scene, pipeline, background, debug=False)
                torch.cuda.synchronize()

                gt = view.original_image[0:3, :, :]
                gt_alpha_mask = view.gt_alpha_mask
                
                # 将结果添加到保存队列（并行I/O）
                save_queue.put((render_pkg.copy(), gt.clone(), gt_alpha_mask.clone(), idx))
                
                # 清理GPU内存
                del render_pkg
                if clear_cache:
                    torch.cuda.empty_cache()
            
            # 批次结束后清理缓存
            if clear_cache:
                torch.cuda.empty_cache()
    
    finally:
        # 等待所有保存任务完成
        save_queue.join()
        
        # 停止保存线程
        for _ in save_threads:
            save_queue.put(None)
        for t in save_threads:
            t.join()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.env_mode, dataset.env_res,
                              dataset.use_sdf, True, True)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 显存优化：根据参数决定是否清理缓存和批处理大小
        clear_cache = getattr(args, 'clear_cache', False)
        batch_size = getattr(args, 'batch_size', 1)
        use_parallel = getattr(args, 'parallel', False)
        num_workers = getattr(args, 'num_workers', 4)
        
        if args.interpolate > 0:
            # 使用dummy图像以节省显存
            use_dummy = not getattr(args, 'full_image_interpolate', False)
            cams = interpolate_camera(scene.getTrainCameras(), args.interpolate, use_dummy_image=use_dummy)
            print(f"生成插值相机: {len(cams)}个, 使用{'占位' if use_dummy else '完整'}图像")
        else:
            cams = scene.getTrainCameras()
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, cams, scene, pipeline, background, 
                       batch_size=batch_size, clear_cache=clear_cache)
             if clear_cache:
                 torch.cuda.empty_cache()

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, pipeline, background,
                       batch_size=batch_size, clear_cache=clear_cache)
             if clear_cache:
                 torch.cuda.empty_cache()

        render_lightings(dataset.model_path, "lighting", scene.loaded_iter, scene, sample_num=1)
        
        # 创建视频（如果使用了插值）
        if args.interpolate > 0 and args.create_video:
            video_dir = os.path.join(dataset.model_path, "videos")
            makedirs(video_dir, exist_ok=True)
            
            if not skip_train:
                # 为训练集渲染结果创建视频
                train_renders_dir = os.path.join(dataset.model_path, "train", "ours_{}".format(scene.loaded_iter), "renders")
                if os.path.exists(train_renders_dir):
                    train_video_path = os.path.join(video_dir, f"train_interpolated_iter_{scene.loaded_iter}.mp4")
                    create_video_from_images(train_renders_dir, train_video_path, fps=args.fps)
                    
                    # 为其他渲染结果（如normal, depth等）创建视频
                    for render_type in ["predicted_normal", "depth", "alpha"]:
                        render_type_dir = os.path.join(dataset.model_path, "train", "ours_{}".format(scene.loaded_iter), render_type)
                        if os.path.exists(render_type_dir):
                            video_path = os.path.join(video_dir, f"train_{render_type}_iter_{scene.loaded_iter}.mp4")
                            create_video_from_images(render_type_dir, video_path, fps=args.fps)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--interpolate", type=int, default=0)
    parser.add_argument("--create_video", action="store_true", help="创建插值渲染的视频")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--clear_cache", action="store_true", help="渲染时清理GPU缓存以节省显存")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小，减少显存使用")
    parser.add_argument("--low_memory", action="store_true", help="低显存模式，自动开启显存优化")
    parser.add_argument("--full_image_interpolate", action="store_true", help="插值时使用完整图像（高显存需求）")
    parser.add_argument("--parallel", action="store_true", help="启用并行渲染（I/O并行）")
    parser.add_argument("--num_workers", type=int, default=4, help="并行保存的线程数")
    args = get_combined_args(parser)
    
    # 低显存模式自动配置
    if args.low_memory:
        args.clear_cache = True
        args.batch_size = max(1, args.batch_size)  # 确保批处理大小至少为1
        print("启用低显存模式: clear_cache=True, batch_size={}".format(args.batch_size))
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)