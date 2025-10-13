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
        dense_score,
        sigma,
        weight_background,
        anchor_means3D,
        anchor_scales,
        anchor_rotations,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacity,
        vi,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        ET_grade,
        if_depth_correct,
        # if_get_gs_skip_ratio
):
    return _RasterizeGaussians.apply(
        dense_score,
        sigma,
        weight_background,
        anchor_means3D,
        anchor_scales,
        anchor_rotations,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacity,
        vi,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        ET_grade,
        if_depth_correct,
        # if_get_gs_skip_ratio
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            dense_score,
            sigma,
            weight_background,
            anchor_means3D,
            anchor_scales,
            anchor_rotations,
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacity,
            vi,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
            ET_grade,
            if_depth_correct,
            # if_get_gs_skip_ratio
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            float(sigma[0]),
            float(weight_background[0]),
            raster_settings.bg,
            anchor_means3D,
            anchor_scales,
            anchor_rotations,
            means3D,
            colors_precomp,
            opacity,
            vi,
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
            float(ET_grade),
            bool(if_depth_correct),
            # if_get_gs_skip_ratio
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(opacity, vi, torch.tensor(ET_grade, dtype=torch.float), torch.tensor(if_depth_correct, dtype=torch.bool), sigma, color, colors_precomp, means3D, scales, rotations,
                                cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        # accum_weights_ptr, accum_weights_count, accum_weights_count
        # Only calcuate for sampling
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        opacity, vi, ET_grade, if_depth_correct, sigma, render_image, colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            float(sigma[0]),
            render_image,
            opacity,
            vi,
            raster_settings.bg,
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
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
            ET_grade,
            if_depth_correct
            )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                dense_score, grad_means2D, grad_colors_precomp, grad_opacity, grad_vi, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_sigma, grad_weight_background = _C.rasterize_gaussians_backward(
                    *args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            dense_score, grad_means2D, grad_colors_precomp, grad_opacity, grad_vi, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_sigma, grad_weight_background = _C.rasterize_gaussians_backward(
                *args)

        if torch.any(torch.isnan(grad_means2D)):
            cpu_args = cpu_deep_copy_tuple(args)
            torch.save(cpu_args, "snapshot_bw.dump")
            raise ValueError("发现 grad_means2D 中有 NaN 值，程序将终止")

        grads = (
            dense_score,
            torch.tensor([grad_sigma.mean()]).to('cuda'),
            torch.tensor([grad_weight_background.mean()]).to('cuda'),
            None, # anchor mean 3D
            None, # anchor scale
            None, # anchor rotation
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacity,
            grad_vi,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None, # ET grade
            None, # if depth correct
            # None, # if get_gs_skip_ratio
        )
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


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

    def forward(
        self,
        dense_score,
        sigma,
        weight_background, 
        anchor_means3D,
        anchor_scales,
        anchor_rotations,
        means3D,
        means2D,
        opacities,
        vi,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        ET_grade=100000.0,  # which means no Early Termination. 
        if_depth_correct=True,
        # if_get_gs_skip_ratio=False,
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            dense_score,
            sigma,
            weight_background,
            anchor_means3D,
            anchor_scales,
            anchor_rotations,
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            vi,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            ET_grade,
            if_depth_correct,
            # if_get_gs_skip_ratio
        )

    def visible_filter(self, means3D, scales=None, rotations=None, cov3D_precomp=None):

        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_aussians_filter(
                means3D,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3D_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.prefiltered,
                raster_settings.debug
            )
        return radii

