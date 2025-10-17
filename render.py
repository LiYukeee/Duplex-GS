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
from utils.system_utils import autoChooseCudaDevice
autoChooseCudaDevice()
import os
from os import makedirs
import torch
import numpy as np
from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from PIL import Image
from utils.general_utils import safe_state, PILtoTorch
from utils.image_utils import psnr
from utils.loss_utils import ssim
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import lpips
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

def testFPS(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
    """
    input: Keep the same input parameters as render_set(...)
    output: the output is a more accurate FPS.
    """
    t_list_len = 200
    warmup_times = 5
    test_times = 10
    t_list = np.array([1.0] * t_list_len)
    step = 0
    fps_list = []
    while True:
        for view in views:
            step += 1
            torch.cuda.synchronize()
            t0 = time.time()
            gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            _ = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask,
                                ape_code=ape_code)
            torch.cuda.synchronize()
            t1 = time.time()
            t_list[step % t_list_len] = t1 - t0

            if step % t_list_len == 0 and step > t_list_len * warmup_times:
                fps = 1.0 / t_list.mean()
                print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
                fps_list.append(fps)
            if step > t_list_len * (test_times + warmup_times):
                # write fps info to a txt file
                with open(os.path.join(model_path, "point_cloud", "iteration_{}".format(iteration), "FPS.txt"), 'w') as f:
                    f.write("Average FPS: {:.5f}\n".format(np.mean(fps_list)))
                    f.write("FPS std: {:.5f}\n".format(np.std(fps_list)))
                print("Average FPS: {:.5f}, FPS std: {:.5f}".format(np.mean(fps_list), np.std(fps_list)))
                return

def render_set(model_path, resolution, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gts_path, exist_ok=True)
    errors_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    makedirs(errors_path, exist_ok=True)
    if show_level:
        render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
        makedirs(render_level_path, exist_ok=True)

    if pipeline.watermark:
        watermark = Image.open('./data/bungeenerf/watermark.png')
        if resolution != 1:
            downscale_ratio = watermark.size[0] / 1600
            new_size = (int(watermark.size[0] / downscale_ratio), int(watermark.size[1] / downscale_ratio))
            del downscale_ratio
        else:
            new_size = watermark.size
        watermark = PILtoTorch(watermark, new_size)
        watermark_alpha = watermark[3:4,:,:].to('cuda')
        watermark = watermark[:3,:,:].to('cuda') * watermark_alpha
        del new_size

    t_list = []
    per_view_dict = {}
    per_view_level_dict = {}
    psnrs = []
    ssims = []
    lpipss = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()

        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
        
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1-t0)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()  
        per_view_dict['{0:05d}'.format(idx)+".png"] = visible_count.item()

        if pipeline.watermark:
            rendering = rendering * (1 - watermark_alpha) + watermark

        gt = view.original_image[0:3, :, :]
        error_map = torch.mean(torch.abs(rendering - gt), dim=0)
        psnrs.append(psnr(rendering, gt).mean())
        ssims.append(ssim(rendering, gt))
        lpipss.append(lpips_fn(rendering, gt).detach())
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(error_map, os.path.join(errors_path, '{0:05d}'.format(idx) + ".png"))

        if show_level:
            for cur_level in range(gaussians.levels):
                gaussians.set_anchor_mask_perlevel(view.camera_center, view.resolution_scale, cur_level)
                voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
                render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
                
                rendering = render_pkg["render"]
                visible_count = render_pkg["visibility_filter"].sum()
                
                # torchvision.utils.save_image(rendering, os.path.join(render_level_path, '{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"))
                per_view_level_dict['{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"] = visible_count.item()

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    psnr_test = torch.tensor(psnrs).mean()
    ssims_test = torch.tensor(ssims).mean()
    lpipss_test = torch.tensor(lpipss).mean()
    print(f'Test FPS: {fps:.5f}')
    print(f'Test PSNR: {psnr_test:.5f}')
    print(f'Test SSIM: {ssims_test:.5f}')
    print(f'Test LPIPS: {lpipss_test:.5f}')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True) 
    if show_level:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count_level.json"), 'w') as fp:
            json.dump(per_view_level_dict, fp, indent=True)

def render_video(n_frames, model_path, name, iteration, views, gaussians, pipeline, background, ape_code):
    fps = 30
    height = views[0].image_height
    width = views[0].image_width
    traj_dir = os.path.join(model_path, 'traj', "ours_{}".format(iteration))
    os.makedirs(traj_dir, exist_ok=True)
    print(f"rendering video to {traj_dir}, n_frames={n_frames}, fps={fps}, height={height}, width={width}")

    from utils.render_utils import generate_path
    import cv2
    cam_traj = generate_path(views, n_frames=n_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(traj_dir, "render_traj_color.mp4"), fourcc, fps, (width, height))

    for view in tqdm(cam_traj, desc="Rendering video"):
        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        rendering = \
        render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)["render"]
        frame = rendering.cpu().permute(1, 2, 0).numpy()
        frame = (frame * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame_bgr)
    video.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, video_flag: bool, show_level : bool, ape_code : int):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_vi_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend,
            dataset.depth_correct, dataset.ET_grade, dataset.anchor_search, dataset.max_sh_degree, dataset.ET_grade_final
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()
        gaussians.plot_levels()
        gaussians.active_sh_degree = gaussians.max_sh_degree
        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if not skip_train:
            render_set(dataset.model_path, dataset.resolution, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, show_level, ape_code)

        if not skip_test:
            render_set(dataset.model_path, dataset.resolution, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level, ape_code)

        if video_flag:
            n_frames = 300
            render_video(n_frames, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, ape_code)

        print("Test FPS:")
        testFPS(dataset.model_path, "test", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,
                show_level, ape_code)

def get_gs_model(path):
    """
    This function reads the checkpoint of the Gaussian model and then returns it the Viewer.
    """
    import sys
    sys.argv = ['render.py', '--model_path', path]
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = model.extract(args)
    iteration = args.iteration
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_vi_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend,
            dataset.depth_correct, dataset.ET_grade, dataset.anchor_search, dataset.max_sh_degree, dataset.ET_grade_final
            )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales, viewer=True)
        gaussians.eval()
        gaussians.plot_levels()
        return gaussians, pipeline

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_video, args.show_level, args.ape)
    
