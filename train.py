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
from gaussian_renderer import render, network_gui, visi_acc_render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state , get_expon_lr_func
import uuid
import kornia
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image, normal2curv, get_edge_aware_distortion_map
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.loss_utils import edge_aware_normal_loss, compute_mutual_axis_loss
from utils.point_utils import depth_to_normal
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = False
except ImportError:
    TENSORBOARD_FOUND = False

def compose_levels_gaussian_for_current_views(opt, gaussians,camera,resolution_scale=[1.0]):
        
    dists_lod = torch.sqrt(((gaussians.get_xyz - camera.camera_center)**2).sum(dim=1)) * camera.resolution_scale  # 计算点到相机距离
    opt.hierachical_depth = torch.linspace(camera.min_dist,camera.max_dist,max(gaussians.levels,3)).cuda()

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
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() 
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

def prune_low_contribution_gaussians(gaussians, cameras, pipe, bg, K=5, prune_ratio=0.01):
    top_list = [None, ] * K
    for i, cam in enumerate(cameras):
        trans= render(cam, gaussians, pipe, bg, record_transmittance=True)
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any():
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]

    contribution = torch.stack(top_list, dim=-1).mean(-1)
    tile = torch.quantile(contribution, prune_ratio)
    prune_mask = contribution < tile
    # 50% 
    # mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()

def get_visi_list(gaussians, viewpoint_stack, pipe, bg):
    out = {}
    gaussian_list = None
    viewpoint_cam = viewpoint_stack.pop()
    v_render = visi_acc_render
    render_pkg = v_render(viewpoint_cam, gaussians, pipe, bg)
    gaussian_list = render_pkg['countgaus']
    
    for i in range(len(viewpoint_stack)):
        # Pick a random Camera
        # prunning
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = v_render(viewpoint_cam, gaussians, pipe, bg)
        gaussians_count = render_pkg["countgaus"].detach()
        gaussian_list += gaussians_count
        
    visi = gaussian_list > 0
        
    out["visi"] = visi
    return out

def get_visi_mask_acc(gaussains, pipe, bg, n=500, up=False, around=True,  viewpoint_stack=None, sample_mode='grid'):
        fullcam = viewpoint_stack
        idx = torch.randint(0, len(fullcam), (n,))
        viewpoint_stack = [fullcam[i] for i in idx]
            
        out = get_visi_list(gaussains, viewpoint_stack, pipe, bg)
        visi = out['visi']
        # inside = self.model.get_inside_gaus_normalized()[0]
        # visi = visi & inside
        
        return visi

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,resolution_scales=dataset.resolution_scales)
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
    all_cameras = scene.getTrainCameras().copy()

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        # camera_center = torch.stack([view.camera_center for view in viewpoint_stack]).mean(dim=0)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # LOSS
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.use_decoupled_appearance:  # L1损失中加入外观模型
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)  
        else:
            Ll1 = l1_loss(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_alpha = render_pkg['rend_alpha']
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_depth = render_pkg['surf_depth']
        surf_normal = render_pkg['surf_normal']

        # 曲率梯度loss  时间与高斯数量作为代价 有提升
        # if iteration < opt.atom_proliferation_until:
        #     Lnormal = edge_aware_normal_loss(gt_image, depth_to_normal(viewpoint_cam, render_pkg["surf_depth"]).permute(2,0,1))
        #     loss += opt.lambda_normal * Lnormal   # lambda = 0.1
        # mask_vis = (rend_alpha > 1e-5)
        # curv_n = normal2curv(surf_normal, mask_vis)  
        # loss_curv = l1_loss(curv_n * 1, 0)
        # curv_loss = 0.005 * loss_curv

        # 法线一致性损失中加入曲率作为权重  暂定无用
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
        # 随图像梯度调整深度失真项 暂定有用
        rend_dist = get_edge_aware_distortion_map(gt_image,rend_dist)
        dist_loss = lambda_dist * (rend_dist).mean()

        # Depth regularization
        Ll1depth_pure = 0.0
        depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["invdepth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # mutual axis loss
        # lambda_mutaxis = 0.001
        # mutaxis_loss = compute_mutual_axis_loss(gaussians.get_scaling)
        # mutaxis_loss *= lambda_mutaxis

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
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None 
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, iteration)
                
                # 不透明度衰减 有用
                if iteration % opt.opacity_reduce_interval == 0 and opt.use_reduce:
                    gaussians.reduce_opacity()
                        
                # 同化
                if  iteration % opt.atom_interval == 0 and iteration < opt.atom_proliferation_until and iteration >= opt.atom_proliferation_begin: 
                    scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                    # 分裂大高斯
                    # gaussians.densify_and_scale_split(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold, opt.densify_scale_factor, scene_mask, N=2, no_grad=True)
                    visi = None
                    levels_lod = compose_levels_gaussian_for_current_views(opt, gaussians, viewpoint_cam)
                    visi = get_visi_mask_acc(gaussians, pipe, background, opt.sample_cams_num, False, True, scene.getTrainCameras().copy(),sample_mode='random')
                    gaussians.atomize(levels_lod,scene_mask,visi)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # else:  # 15000次迭代后是否需要措施（修剪？致密化？）
                # if iteration % 1000 == 0 and iteration <= 25000:
                #     scene_mask, scene_center = culling(gaussians.get_xyz, scene.getTrainCameras())
                #     visi = get_visi_mask_acc(gaussians, pipe, background, opt.sample_cams_num, False, True, scene.getTrainCameras().copy(),sample_mode='random')
                #     gaussians.atomize_last(scene_mask, visi)

            # 去除低贡献度高斯
            # if iteration <= opt.iterations and iteration % opt.contribution_prune_interval == 0:
            #     prune_low_contribution_gaussians(gaussians, all_cameras, pipe, background, K=5, prune_ratio=opt.contribution_prune_ratio)
 
            # Optimizer step 
            if iteration < opt.iterations:
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