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
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# for debug
import os
import torchvision.transforms as transforms
# os.makedirs("outputs/image", exist_ok=True)
step = 0


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False,  ape_code=-1):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    grid_offsets = pc.get_offset[visible_mask]
    # grid_rot = pc.get_rotation[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)
    ############################# opacity #############################
    ######## mlp ########
    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])

    
    ######## no mlp ########
    # neural_opacity = pc.get_gs_opacity[visible_mask]
    # # opacity mask generation
    # neural_opacity = neural_opacity.reshape([-1, 1])
    # mask = (neural_opacity > 0.0)
    # mask = mask.view(-1)
    
    # opacity = neural_opacity[mask]
    
    ############################# color #############################
    ######## mlp ########
    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]
    ######## no mlp ########
    
    
    ############################# cov and scale #############################
    ######## mlp ########
    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    
    ######## no mlp ########
    # scale = pc.get_gs_scale[visible_mask]
    # rot = pc.get_gs_rot[visible_mask]
    # scale = scale.reshape([-1, 3])
    # rot = rot.reshape([-1, 4])
    # scale_rot = torch.cat((scale, rot), dim=-1)
    
    ############################# vi #############################
    ######## mlp ########
    # mlp vi
    # if pc.add_vi_dist:
    #     neural_vi = pc.get_vi_mlp(cat_local_view)
    # else:
    #     neural_vi = pc.get_vi_mlp(cat_local_view_wodist)
    # vi = neural_vi.repeat([1,pc.n_offsets]).reshape([-1, 1])

    ######## anchor sh vi ########
    shvi = pc.get_anchor_sh_vi[visible_mask]
    neural_vi = eval_sh(pc.active_sh_degree, shvi, ob_view)
    vi = neural_vi.repeat_interleave(pc.n_offsets, dim=0)
    vi = torch.sigmoid(vi)

    ############################# end #############################
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, neural_vi], dim=-1)
    # masked = concatenated_all[mask]
    # scaling_repeat, repeat_anchor, color, scale_rot, offsets, vi = masked.split([6, 3, 3, 7, 3, 1], dim=-1)

    scaling_repeat, repeat_anchor = concatenated_repeated[:, 0:3], concatenated_repeated[:, 3:]
    # post-process cov
    # scaling_ratio_max = (anchor_search_scale / anchor_scale - offset_distance) / (gs_render_scale / gs_scale)
    scaling_ratio_max = (2 - torch.sqrt(torch.sum(offsets.detach()**2, dim=1, keepdim=True))) / 3
    scaling_ratio = torch.sigmoid(scale_rot[:,:3]).clamp(max=scaling_ratio_max)
    scaling = scaling_repeat[:,0:3] * scaling_ratio

    # factor = 0.3
    rot = pc.rotation_activation(scale_rot[:,3:7])
    # rot_dev = torch.tanh(scale_rot[:,3:7])
    # rot = pc.rotation_activation(rot_repeat + rot_dev)

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,0:3]
    xyz = repeat_anchor + offsets 
    
    anchor_means3D = anchor
    # anchor_scales = grid_scaling[:, -3:].unsqueeze(1)
    # anchor_scales = grid_scaling[:, 0:3].max(dim=1)[0].unsqueeze(1)
    anchor_scales = grid_scaling[:, 0:3]

    # set weighted average to be quaternions of cell
    weights = (neural_opacity.detach() * vi.detach()).clamp(min=0.0).reshape([-1, pc.n_offsets])
    anchor_rotations = pc.rotation_activation(
        (rot.detach().reshape([-1, pc.n_offsets, 4]) * weights.unsqueeze(dim=2)).mean(dim=1)
    )
    # weights = (neural_opacity * vi).clamp(min=0.0).reshape([-1, pc.n_offsets])
    # anchor_rotations = pc.rotation_activation(
    #     (rot.reshape([-1, pc.n_offsets, 4]) * weights.unsqueeze(dim=2)).mean(dim=1)
    # )

    if is_training:
        pc._rotation[visible_mask] = pc.rotation_activation(anchor_rotations + pc._rotation[visible_mask])
        return anchor_means3D, anchor_scales, anchor_rotations, xyz, color, neural_opacity, scaling, rot, vi, neural_vi
    else:
        return anchor_means3D, anchor_scales, anchor_rotations, xyz, color, neural_opacity, scaling, rot, vi

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, retain_grad=False, ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    is_training = pc.get_color_mlp.training
        
    if is_training:
        anchor_means3D, anchor_scales, anchor_rotations, xyz, color, opacity, scaling, rot, vi, anchor_vi = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        anchor_means3D, anchor_scales, anchor_rotations, xyz, color, opacity, scaling, rot, vi = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, ape_code=ape_code)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    dense_score = torch.zeros_like(opacity, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
            dense_score.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # sortfree rasterizer
    rendered_image, radii = rasterizer(
        sigma = pc.get_sigma,
        weight_background=pc.get_wbg,
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        vi=vi,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None,
        if_depth_correct = pc.depthcorrect
        )
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    #     dense_score = dense_score,
    #     sigma = pc.get_sigma,
    #     weight_background = pc.get_wbg,
    #     anchor_means3D = anchor_means3D,
    #     anchor_scales = anchor_scales,
    #     anchor_rotations=anchor_rotations,
    #     means3D = xyz,
    #     means2D = screenspace_points,
    #     shs = None,
    #     colors_precomp = color,
    #     opacities = opacity,
    #     vi = vi,
    #     scales = scaling,
    #     rotations = rot,
    #     cov3D_precomp = None,
    #     ET_grade = pc.ET_grade,
    #     if_depth_correct = pc.depthcorrect
    #     )

    # global step
    # step += 1
    # if step % 200 == 0:
    #     image = rendered_image
    #     to_pil_image = transforms.ToPILImage()
    #     image = to_pil_image(image)
    #     filename = "outputs/image/{}-{}.png".format(step, pc.ET_grade)
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     image.save(filename)
    #
    #     print("The proportion of skipped GS: {:.3f}%".format(100 * (1 - num_render_gs.float().mean() / num_val_gs.float().mean())))
    #     skipped_GS_proportion = 1.0 - num_render_gs.float() / num_val_gs.float()
    #     image = to_pil_image(skipped_GS_proportion)
    #     filename = "outputs/image/{}-{}-skipped_GS_proportion.png".format(step, pc.ET_grade)
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     image.save(filename)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": (opacity>0.0).view(-1),
                "neural_opacity": opacity,
                "anchor_vi": anchor_vi,
                "scaling": scaling,
                "dense_score": dense_score,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor[pc._anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling[pc._anchor_mask]
        rotations = pc.get_rotation[pc._anchor_mask]

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,0:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii_pure > 0
    return visible_mask
