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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):  
    if raster_settings.r_type == 1:
        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
        )
    elif raster_settings.r_type == 2:
        return _RasterizeGaussians.forward_visi_acc(
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
        )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.record_transmittance,
            raster_settings.debug
        )

        if raster_settings.r_type == 2:
            args = args + (raster_settings.r_type,)
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    gaussians_count, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                gaussians_count, num_rendered, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
        else:
            # Invoke C++/CUDA rasterizer
            if raster_settings.debug:
                cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
                try:
                    num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, transmittance, num_covered_pixels = _C.rasterize_gaussians(*args)
                except Exception as ex:
                    torch.save(cpu_args, "snapshot_fw.dump")
                    print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                    raise ex
            else:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer, transmittance, num_covered_pixels= _C.rasterize_gaussians(*args)

            if raster_settings.record_transmittance:
                return transmittance, num_covered_pixels, radii

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth
    
    @staticmethod
    def forward_visi_acc(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        assert(raster_settings.r_type == 2)
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.r_type
        )
        # gaussians_count, important_score, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = None, None, None, None, None, None, None, None
        # Invoke C++/CUDA rasterizer
        # TODO(Kevin): pass the count in, but the output include a count list 
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                gaussians_count, num_rendered, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            gaussians_count, num_rendered, radii, geomBuffer, binningBuffer, imgBuffer = _C.visi_gaussians_acc(*args)
            # gaussians_count, _, num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.count_gaussians(*args)
        
        return gaussians_count, radii 


    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        assert not raster_settings.record_transmittance, 'should not execute backward for calculate transmittance'
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    record_transmittance: bool
    debug : bool
    r_type : int

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

