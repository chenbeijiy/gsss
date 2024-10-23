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

import os
import torch
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import kornia
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.loss_utils import edge_aware_normal_loss
from utils.point_utils import depth_to_normal
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def compute_projection_scale_and_distance(lod_scaling_limit, fovx, fovy, width, height, pixel_size=1.0):
    """
    Calculate the distance at which the smallest Gaussian scale appears as one pixel.
    
    Args:
    - scale (torch.Tensor): The scale of the Gaussian in 3D space, assumed to be in the same unit as the focal length.
    - fovx (float): Horizontal field of view in radians.
    - fovy (float): Vertical field of view in radians.
    - width (int): Width of the image in pixels.
    - height (int): Height of the image in pixels.
    - pixel_size (float): The size of a pixel in the same unit as the focal length. Typically set to 1.0 for unit consistency in image processing. (= screensize)

    Returns:
    - float: The maximum distance where the smallest dimension appears as one pixel.
    """
        
    # Compute the focal lengths from field of view
    focal_length_x = width / (2.0 * math.tan(fovx / 2.0))
    focal_length_y = height / (2.0 * math.tan(fovy / 2.0))

    # Use the smaller of the two focal lengths to ensure visibility in both dimensions
    focal_length = min(focal_length_x, focal_length_y)
    # calculate the distance where the 2D projection of the scale constraint equals the predefined screensize threshold (Sec 4.4 eq.6)
    max_distance = focal_length * lod_scaling_limit / pixel_size
    
    return max_distance

def compose_hlod_gaussian_for_all_views(opt,gaussians, view, camera_center,pixel_size=1.0):
        
    scaling_ratio = gaussians.lod_scaling_ratio
    
    # lod_scaling_limit = gaussians.lod_scaling_limit
    lod_scaling_limit = 0.003125
    lod_max_dist_upper_bound = compute_projection_scale_and_distance(lod_scaling_limit, view.FoVx, view.FoVy, view.image_width, view.image_height, pixel_size=pixel_size)
    for i in range(1,4):
        opt.hierachical_depth.append(lod_max_dist_upper_bound)
        lod_max_dist_upper_bound *= scaling_ratio
    
    dists_lod = torch.sqrt(((gaussians.get_xyz - camera_center)**2).sum(dim=1))  # 即为点到相机中心点距离

    levels_lod = torch.where(dists_lod < opt.hierachical_depth[0], 0,  # 区间 1
               torch.where(dists_lod < opt.hierachical_depth[1], 1,  # 区间 2
               torch.where(dists_lod < opt.hierachical_depth[2], 2,  # 区间 3
               3)))  # 区间 4
    
    return levels_lod

def compose_hlod_gaussian_for_all_views1(opt, gaussians,camera_center,resolution_scale=[1.0]):
        
    dists_lod = torch.sqrt(((gaussians.get_xyz - camera_center)**2).sum(dim=1))  # 即为点到相机距离
    opt.hierachical_depth = torch.linspace(gaussians.min_dist,gaussians.max_dist,max(gaussians.levels,3)).cuda()
    # pred_level = torch.log2(gaussians.standard_dist/dists_lod)
    # pred_level = torch.min(torch.max(pred_level,0),gaussians.levels)

    levels_lod = torch.searchsorted(opt.hierachical_depth, dists_lod, right=True)
    
    return levels_lod

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

def culling(xyz, cams, expansion=2):
    cam_centers = torch.stack([c.camera_center for c in cams], 0).to(xyz.device)
    span_x = cam_centers[:, 0].max() - cam_centers[:, 0].min()
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() # smallest span
    span_z = cam_centers[:, 2].max() - cam_centers[:, 2].min()

    scene_center = cam_centers.mean(0)

    span_x = span_x * expansion
    span_y = span_y * expansion
    span_z = span_z * expansion

    x_min = scene_center[0] - span_x / 2
    x_max = scene_center[0] + span_x / 2

    y_min = scene_center[1] - span_y / 2
    y_max = scene_center[1] + span_y / 2

    z_min = scene_center[2] - span_x / 2
    z_max = scene_center[2] + span_x / 2


    valid_mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
                 (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
                 (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    # print(f'scene mask ratio {valid_mask.sum().item() / valid_mask.shape[0]}')

    return valid_mask, scene_center

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for idx, camera in enumerate(scene.getTrainCameras()):
        camera.idx = idx
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        camera_center = torch.stack([view.camera_center for view in viewpoint_stack]).mean(dim=0)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # LOSS
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)
        else:
            Ll1 = l1_loss(image, gt_image)
        
        #尺度loss
        max_values, _ = torch.max(gaussians.get_scaling[:], dim=1)
        scale_reg = 0.01
        max_values = max_values[max_values > scale_reg]
        lscale = torch.sum(max_values)
        ld_scale = 0.0005

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) # + ld_scale * lscale
        
        # if iteration < opt.atom_proliferation_until: 
        #     Lnormal = edge_aware_normal_loss(gt_image, depth_to_normal(viewpoint_cam, render_pkg["surf_depth"]).permute(2,0,1))
        #     loss += opt.lambda_normal * Lnormal   # lambda = 0.1

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_depth = render_pkg['surf_depth']
        surf_normal = render_pkg['surf_normal']

        # depth_map = surf_depth[0]
        # if opt.depth_grad_thresh > 0:
        #     depth_map_for_grad = depth_map[None, None]
        #     sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
        #     sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)

        #     depth_map_for_grad = F.pad(depth_map_for_grad, pad=(1, 1, 1, 1), mode="replicate")
        #     depth_grad_x = F.conv2d(depth_map_for_grad, sobel_kernel_x) / 3
        #     depth_grad_y = F.conv2d(depth_map_for_grad, sobel_kernel_y) / 3
        #     depth_grad_mag = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)
        #     depth_grad_weight = (depth_grad_mag < opt.depth_grad_thresh).float()
        #     if opt.depth_grad_mask_dilation > 0:
        #         mask_di = opt.depth_grad_mask_dilation
        #         depth_grad_weight = -1 * F.max_pool2d(-1 * depth_grad_weight, mask_di * 2 + 1, stride=1, padding=mask_di)
        #     depth_grad_weight = depth_grad_weight.squeeze().detach()
        #     normal_error = (1 - (rend_normal * surf_normal).sum(dim=0)) * depth_grad_weight
        #     normal_error = normal_error[None]
        # else:
        #     normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]

        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # smooth loss
        # lambda_smooth = 1.0
        # smooth_loss = kornia.losses.inverse_depth_smoothness_loss(surf_depth.unsqueeze(0), gt_image.unsqueeze(0))
        # smooth_loss = lambda_smooth * smooth_loss
        

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict) 

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            # iteration = 15000
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, iteration)
                    # scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                    # gaussians.densify_and_scale_split(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, opt.max_screen_size, opt.densify_scale_factor, scene_mask, N=3, no_grad=True)

                # 同化
                if  iteration % opt.atom_interval == 0 and iteration < opt.atom_proliferation_until and iteration > opt.atom_proliferation_begin: 
                    levels_lod = compose_hlod_gaussian_for_all_views1(opt, gaussians, viewpoint_cam.camera_center)
                    scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                    gaussians.atomize(levels_lod,scene_mask)

                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    # gaussians.reset_opacity_spike()
            # else:
            #     if iteration % opt.densification_interval == 0 or iteration == 29999:
            #         gaussians.spike_prune(gaussians.Vth_opa.item())
 

            # Optimizer step
            if iteration < opt.iterations:
            # if iteration < 7000:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

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

        torch.cuda.empty_cache()

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
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")