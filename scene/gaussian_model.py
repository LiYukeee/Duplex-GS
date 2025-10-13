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

import time
from datetime import timedelta
import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from einops import repeat
import math

class LearnableScaleLayer(nn.Module):
    def __init__(self, initial_value=1.0):
        super(LearnableScaleLayer, self).__init__()
        self.scale_factor = nn.Parameter(torch.tensor(initial_value))

    def forward(self, x):
        return x * self.scale_factor

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        # gs parameter
        self.gs_opacity_activation = torch.sigmoid
        self.gs_scaling_activation = torch.exp
        self.gs_rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 fork: int=2,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 add_vi_dist : bool = False,  # newly added
                 add_level: bool = False,
                 visible_threshold: float = -1,
                 dist2level: str = 'round',
                 base_layer: int = 10,
                 progressive: bool = True,
                 extend: float = 1.1,
                 depth_correct: bool = False,  # newly added
                 ET_grade: float = 100000.0,  # newly added
                 if_anchor_search: bool = False,  # newly added
                 max_sh_degree: int = 3,  # newly added
                 ET_grade_final: float = 4.0,  # newly added
                 ):

        self.feat_dim = feat_dim
        self.view_dim = 3
        self.n_offsets = n_offsets
        self.fork = fork
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_vi_dist = add_vi_dist
        self.add_level = add_level
        self.progressive = progressive

        # Octree
        self.sub_pos_offsets = torch.tensor([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float().cuda()
        self.extend = extend
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer
        
        self.start_step = 0
        self.end_step = 0

        # parameters for anchor
        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)  # decoded as color
        self._anchor_size = torch.empty(0)
        self._rotation = torch.empty(0)
        # # used for anchor pruning
        self.opacity_accum = torch.empty(0)
        self.vi_accum = torch.empty(0)  # newly added
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)
        self.dense_score = torch.empty(0)

        ### sortfree
        self._sigma = torch.empty(0)
        self._wbg = torch.empty(0)  # weight background
        self.active_sh_degree = 0
        self.max_sh_degree = max_sh_degree
        ### sortfree anchor sh vi
        self._features_anchor_dv = torch.empty(0)
        self._features_anchor_rv = torch.empty(0)

        # additional rasterization parameter
        self.if_anchor_search = if_anchor_search
        self.depthcorrect = depth_correct
        self.ET_grade = ET_grade
        self.ET_grade_final = ET_grade_final

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.level_dim = 1 if self.add_level else 0
        
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(self.view_dim+self.level_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()

        # sortfree mlp vi
        self.vi_dist_dim = 1 if self.add_vi_dist else 0
        # self.mlp_vi = nn.Sequential(
        #         nn.Linear(self.feat_dim+self.view_dim+self.vi_dist_dim+self.level_dim, self.feat_dim),
        #         nn.ReLU(True),
        #         nn.Linear(self.feat_dim, self.n_offsets),
        #         nn.ReLU(True)
        #     ).cuda()
        self.mlp_vi = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.vi_dist_dim+self.level_dim, int(self.feat_dim)),
                nn.ReLU(True),
                nn.Linear(int(self.feat_dim), 1),
                nn.Sigmoid()
                # LearnableScaleLayer()
            ).cuda()
            
        self.mlp_opacity = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, self.n_offsets),
                nn.Tanh()
            ).cuda()
        
        self.mlp_cov = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 7*self.n_offsets),
            ).cuda()
        
        self.mlp_color = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        self.mlp_vi.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        self.mlp_vi.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def capture(self):  # implemented to interrupt & continue training (not complete yet)
        return (
            self._anchor,
            self._level,
            self._offset,
            self._local,
            # self._scaling,
            self._rotation,
            # self._opacity,
            # anchor sh vi
            self._features_anchor_dv,
            self._features_anchor_rv,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):  # implemented to interrupt & continue training (not complete yet)
        (self.active_sh_degree, 
        self._anchor, 
        self._level,
        self._offset,
        self._local,
        # self._scaling,
        self._rotation,
        # self._opacity,
        # anchor sh vi
        self._features_anchor_dv,
        self._features_anchor_rv,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._anchor_size)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_offset(self):
        return torch.clamp(self._offset, min=-1.0, max=1.0)

    @property
    def get_level(self):
        return self._level
    
    @property
    def get_extra_level(self):
        return self._extra_level
        
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_anchor_feat(self):
        return self._anchor_feat
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity   

    @property
    def get_cov_mlp(self):
        return self.mlp_cov
    
    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_vi_mlp(self):
        return self.mlp_vi
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_sigma(self):
        return self._sigma
    
    @property
    def get_wbg(self):
        return torch.relu(self._wbg) + 0.1
    
    @property
    def get_anchor_sh_vi(self):
        features_dv = self._features_anchor_dv
        features_rv = self._features_anchor_rv
        shvi = torch.cat((features_dv, features_rv), dim=-1)
        return shvi
    
    
    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)    

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        self.coarse_intervals = []
        num_level = self.levels - 1 - self.init_level
        if num_level > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, dist_ratio)
                dist_min = torch.quantile(dist, 1 - dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.standard_dist = dist_max
        if levels == -1:
            self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels/2)
        else:
            self.init_level = init_level
        self.view_levels = self.levels
            
    def octree_sample(self, data, init_pos):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda() 
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - init_pos) / cur_size), dim=0) * cur_size + init_pos
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building octree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, points, spatial_lr_scale, logger=None):
        self.spatial_lr_scale = spatial_lr_scale
        box_min = torch.min(points)*self.extend
        box_max = torch.max(points)*self.extend
        box_d = box_max - box_min
        if self.base_layer < 0:
            default_voxel_size = 0.02
            self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
        self.octree_sample(points, self.init_pos)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)

        print(f'Branches of Tree: {self.fork}')
        print(f'Base Layer of Tree: {self.base_layer}')
        print(f'Visible Threshold: {self.visible_threshold}')
        print(f'Appearance Embedding Dimension: {self.appearance_dim}') 
        print(f'LOD Levels: {self.levels}')
        print(f'Initial Levels: {self.init_level}')
        print(f'Initial Voxel Number: {self.positions.shape[0]}')
        print(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        print(f'Max Voxel Size: {self.voxel_size}')
        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Base Layer of Tree: {self.base_layer}')
        logger.info(f'Visible Threshold: {self.visible_threshold}')
        logger.info(f'Appearance Embedding Dimension: {self.appearance_dim}')
        logger.info(f'LOD Levels: {self.levels}')
        logger.info(f'Initial Levels: {self.init_level}')
        logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')
        logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')
        logger.info(f'Max Voxel Size: {self.voxel_size}')

        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(self.positions).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((self.positions.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.5 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(self.positions.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        # self._anchor_size = nn.Parameter(scales.requires_grad_(True))
        self._anchor_size = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._rotation = rots
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        
        # SortFree
        self._sigma = nn.Parameter(torch.tensor([10.0], dtype=torch.float32, device='cuda').requires_grad_(True))
        self._wbg = nn.Parameter(torch.tensor([0.02], dtype=torch.float32, device='cuda').requires_grad_(True))
        
        # anchor sh vi
        self.SH_C0 = 0.28209479177387814
        features_anchor_v = torch.zeros((self.positions.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_anchor_v[:, :, 0] = 0.5 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda") / self.SH_C0
        self._features_anchor_dv = nn.Parameter(features_anchor_v[:,:,0:1].contiguous().requires_grad_(True))
        self._features_anchor_rv = nn.Parameter(features_anchor_v[:,:,1:].contiguous().requires_grad_(True))

    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")
        
        return int_level

    def weed_out(self, anchor_positions, anchor_levels):
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)   
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        # anchor_pos = self._anchor + (self.voxel_size/2) / (float(self.fork) ** self._level)
        anchor_pos = self._anchor
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        
        is_training = self.get_color_mlp.training
        if self.progressive and is_training:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        int_level = self.map_to_int_level(pred_level, coarse_index - 1)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        anchor_pos = self._anchor + (self.voxel_size/2) / (float(self.fork) ** self._level)
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.vi_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.dense_score = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._anchor_size], 'lr': training_args.scaling_lr, "name": "anchor_size"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "anchor_rot"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            
            # sortfree
            # learning rate for sigma, wbg and vi
            {'params': [self._sigma], 'lr': training_args.sigma_lr * self.spatial_lr_scale, "name": "sigma"},
            {'params': [self._wbg], 'lr': training_args.wbg_lr_init, "name": "wbg"},
            {'params': self.mlp_vi.parameters(), 'lr': training_args.mlp_vi_lr_init, "name": "mlp_vi"},
            
            # anchor sh vi
            {'params': [self._features_anchor_dv], 'lr': training_args.shvi, "name": "f_anchor_dv"},
            {'params': [self._features_anchor_rv], 'lr': training_args.shvi, "name": "f_anchor_rv"},
        ]
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        
        self.mlp_vi_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_vi_lr_init,
                                                    lr_final=training_args.mlp_vi_lr_final,
                                                    lr_delay_mult=training_args.mlp_vi_lr_delay_mult,
                                                    max_steps=training_args.mlp_vi_lr_max_steps)
        
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        
        # sortfree
        self.sigma_scheduler_args = get_expon_lr_func(lr_init=training_args.sigma_lr*self.spatial_lr_scale,
                                            lr_final=training_args.sigma_lr_final*self.spatial_lr_scale,
                                            lr_delay_mult=training_args.sigma_lr_delay_mult,
                                            max_steps=training_args.sigma_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            # sortfree
            if param_group["name"] == "sigma":
                lr = self.sigma_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_vi":
                lr = self.mlp_vi_scheduler_args(iteration)
                param_group['lr'] = lr
            
    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        l.append('info')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._anchor_size[:,:3].shape[1]):
            l.append('anchor_size_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('anchor_rot_{}'.format(i))
        # shvi
        # for i in range(self.n_offsets):
        #     l.append('shd_{}'.format(i))
        # for i in range(self.n_offsets * ((self.max_sh_degree + 1) ** 2 - 1)):
        #     l.append('shv_{}'.format(i))
        # anchor-shvi
        for i in range(1):
            l.append('anchor_shd_{}'.format(i))
        for i in range((self.max_sh_degree + 1) ** 2 - 1):
            l.append('anchor_shv_{}'.format(i))
        # gs parameter
        
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist
        infos[2, 0] = self._sigma
        infos[3, 0] = self._wbg
        infos[4, 0] = self.ET_grade

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        anchor_size = self._anchor_size[:,:3].detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()  # anchor rot (fixed, unlearnable, no need to save locally)
        
        # sortfree
        features_anchor_dv = self._features_anchor_dv.view(self._features_anchor_dv.shape[0], -1).detach().cpu().numpy()
        features_anchor_rv = self._features_anchor_rv.view(self._features_anchor_rv.shape[0], -1).detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, anchor_size, rots, features_anchor_dv, features_anchor_rv), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def plot_levels(self):
        for level in range(self.levels):
            level_mask = (self._level == level).squeeze(dim=1)
            print(f'Level {level}: {torch.sum(level_mask).item()}, Ratio: {torch.sum(level_mask).item()/self._level.shape[0]}')

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        
        levels = np.asarray(plydata.elements[0]["level"])[... ,np.newaxis].astype(int)
        extra_levels = np.asarray(plydata.elements[0]["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata.elements[0]["info"][0]).float()
        self.standard_dist = torch.tensor(plydata.elements[0]["info"][1]).float()

        # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        anchor_size_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("anchor_size_")]
        anchor_size_names = sorted(anchor_size_names, key = lambda x: int(x.split('_')[-1]))
        anchor_size = np.zeros((anchor.shape[0], len(anchor_size_names)))
        for idx, attr_name in enumerate(anchor_size_names):
            anchor_size[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("anchor_rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        anchor_shd_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("anchor_shd_")]
        anchor_shd_names = sorted(anchor_shd_names, key = lambda x: int(x.split('_')[-1]))
        anchor_shd = np.zeros((anchor.shape[0], len(anchor_shd_names)))
        for idx, attr_name in enumerate(anchor_shd_names):
            anchor_shd[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        anchor_shd = anchor_shd.reshape((offsets.shape[0], 1, -1))

        anchor_shv_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("anchor_shv_")]
        anchor_shv_names = sorted(anchor_shv_names, key = lambda x: int(x.split('_')[-1]))
        anchor_shv = np.zeros((anchor.shape[0], len(anchor_shv_names)))
        for idx, attr_name in enumerate(anchor_shv_names):
            anchor_shv[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        anchor_shv = anchor_shv.reshape((offsets.shape[0], 1, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._anchor_size = nn.Parameter(torch.tensor(anchor_size, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = torch.max(self._level) - torch.min(self._level) + 1

        # sortfree
        self._sigma = nn.Parameter(torch.tensor([plydata.elements[0]["info"][2]]).float().cuda().requires_grad_(True))
        self._wbg = nn.Parameter(torch.tensor([plydata.elements[0]["info"][3]]).float().cuda().requires_grad_(True))

        self.ET_grade = plydata.elements[0]["info"][4]
        # sh vi
        self._features_anchor_dv = nn.Parameter(torch.tensor(anchor_shd, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_anchor_rv = nn.Parameter(torch.tensor(anchor_shv, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'sigma' in group['name'] or \
                'wbg' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, dense_score, opacity, vi, update_filter, offset_selection_mask, anchor_visible_mask):
        """
        opacity: gs opacity [valid_anchor * 10, 1]
        vi: anchor vi [valid_anchor, 1]
        update_filter: anchor_radii > 0 [valid_anchor, 1]
        offset_selection_mask: gs_opacity > 0 [valid_anchor * 10, 1]
        anchor_visible_mask: anchors filtered based on LOD and the view frustum. [num_anchor, 1], anchor_visible_mask.sum() = valid_anchor
        """
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update vi stats
        temp_vi = vi.clone().detach()
        temp_vi[temp_vi<=0] = 0
        self.vi_accum[anchor_visible_mask] += temp_vi
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        update_filter = update_filter.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)  # anchor_visible_mask -> gaussian_visible_mask
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        update_filter = torch.logical_and(update_filter, offset_selection_mask)
        combined_mask[anchor_visible_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
        
        temp_dense_score = dense_score.grad[update_filter]
        self.dense_score[combined_mask] = torch.max(self.dense_score[combined_mask], temp_dense_score)
        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'sigma' in group['name'] or \
                'wbg' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        # self._opacity = optimizable_tensors["opacity"]
        self._anchor_size = optimizable_tensors["anchor_size"]
        # self._rotation = optimizable_tensors["anchor_rot"]
        self._level = self._level[valid_points_mask]    
        self._extra_level = self._extra_level[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]
        
        # anchor sh vi
        self._features_anchor_dv = optimizable_tensors["f_anchor_dv"]
        self._features_anchor_rv = optimizable_tensors["f_anchor_rv"]

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates
    
    def anchor_growing(self, iteration, grads, dense_score, threshold, dense_score_threshold, update_ratio, extra_ratio, extra_up, offset_mask, use_dense_score, prune_mask_big_scale):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        dense_score[~offset_mask] = 0.0
        update_value = self.fork ** update_ratio
        ########## mask from grad ##########
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        if use_dense_score:
            ########## mask from Dense Score ##########
            anchor_dense_score = torch.sum(dense_score.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        
        for cur_level in range(self.levels):
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            ########## mask from grad ##########
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)  # Get the candidate anchors meeting the gradients requirements
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)
            if use_dense_score:
                ########## mask from Dense Score ##########
                # update threshold
                cur_threshold = dense_score_threshold * (update_value ** cur_level)
                ds_threshold = cur_threshold * update_value
                extra_threshold = cur_threshold * extra_ratio
                
                candidate_mask_dense_score = (dense_score >= cur_threshold) & (dense_score < ds_threshold)
                candidate_ds_mask_dense_score = (dense_score >= ds_threshold)
                candidate_extra_mask_dense_score = (anchor_dense_score >= extra_threshold)
                
                # combine grad's choice and dense score's choice
                candidate_mask = torch.logical_or(candidate_mask, candidate_mask_dense_score)
                candidate_ds_mask = torch.logical_or(candidate_ds_mask, candidate_ds_mask_dense_score)
                candidate_extra_mask = torch.logical_or(candidate_extra_mask, candidate_extra_mask_dense_score)

            length_inc = self.get_anchor.shape[0] - init_length
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)   
            
            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets) # Get the candidate anchors within this level
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()    

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self.get_offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            # all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            if cur_level == 0:   # grow by spliting big anchors (grow to level-0)
                xyz_big_scale = self.get_anchor[prune_mask_big_scale].unsqueeze(dim=1) + self.get_offset[
                    prune_mask_big_scale] * self.get_scaling[prune_mask_big_scale, :3].unsqueeze(dim=1)
                selected_xyz = torch.cat([selected_xyz, xyz_big_scale.view([-1, 3])], dim=0)
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size+self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size).int()
                selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
                selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size).int()
                selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:

                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                if cur_level == 0:
                    big_scale_feat = self._anchor_feat[prune_mask_big_scale].unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])
                    new_feat = torch.cat([new_feat, big_scale_feat], dim=0)
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float, device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)

                new_scaling = torch.ones_like(candidate_anchor).float().cuda()*cur_size # *0.05
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).float().cuda()*ds_size # *0.05
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_anchor_size = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])

                # anchor sh vi
                self.SH_C0 = 0.28209479177387814
                features_anchor_v = torch.zeros((candidate_anchor.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
                features_anchor_v[:, :, 0] = 0.5 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda") / self.SH_C0
                features_anchor_v_ds = torch.zeros((candidate_anchor_ds.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
                features_anchor_v_ds[:, :, 0] = 0.5 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda") / self.SH_C0
                new_features_anchor_dv = torch.cat([features_anchor_v[:, :, :1], features_anchor_v_ds[:, :, :1]])
                new_features_anchor_rv = torch.cat([features_anchor_v[:, :, 1:], features_anchor_v_ds[:, :, 1:]])

                # re-initialize candidate GSs
                ## offset
                self._offset[candidate_mask.view([-1,self.n_offsets])] = torch.zeros([candidate_mask.sum(), 3]).float().cuda()
                self._offset[candidate_ds_mask.view([-1, self.n_offsets])] = torch.zeros([candidate_ds_mask.sum(), 3]).float().cuda()

                d = {
                    # anchor parameters
                    "anchor": new_anchor,
                    "anchor_size": new_anchor_size,
                    "anchor_feat": new_feat,
                    # "anchor_rot": new_rotation,
                    # GS parameters
                    "offset": new_offsets,
                    "f_anchor_dv": new_features_anchor_dv,
                    "f_anchor_rv": new_features_anchor_rv,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                temp_vi_accum = torch.cat([self.vi_accum, torch.zeros([new_anchor.shape[0], 1], device='cuda').float()], dim=0)
                del self.vi_accum
                self.vi_accum = temp_vi_accum

                torch.cuda.empty_cache()

                # learnable parameters
                # # anchor parameter
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._anchor_size = optimizable_tensors["anchor_size"]
                # self._rotation = optimizable_tensors["anchor_rot"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                # anchor sh vi
                self._features_anchor_dv = optimizable_tensors["f_anchor_dv"]
                self._features_anchor_rv = optimizable_tensors["f_anchor_rv"]

                # other un-learnable parameters
                self._rotation = torch.cat([self._rotation, new_rotation], dim=0)
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)

    def adjust_anchor(self, voxel_visible_mask, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, dense_score_threshold=0.0007, update_ratio=0.5, extra_ratio=4.0, extra_up=0.25, min_opacity=0.005, min_vi=0.01, use_dense_score=False, max_anchor_size=3.0):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        start_anchor = self.get_anchor.shape[0]
        prune_mask_big_scale = (self.get_scaling[:, 0:3].max(dim=1)[0] > max_anchor_size)
        self.anchor_growing(iteration, grads_norm, self.dense_score.view(-1), grad_threshold, dense_score_threshold, update_ratio, extra_ratio, extra_up, offset_mask, use_dense_score, prune_mask_big_scale)
        end_anchor = self.get_anchor.shape[0]
        print(f'Added {end_anchor - start_anchor} anchors')

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        self.dense_score[offset_mask] = 0
        padding_dense_score = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.dense_score.shape[0], 1],
                                           dtype=torch.float, 
                                           device=self.dense_score.device)
        self.dense_score = torch.cat([self.dense_score, padding_dense_score], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        prune_mask_vi = (self.vi_accum < min_vi*self.anchor_demon).squeeze(dim=1)
        print("opacity prun: ", int(prune_mask.sum()))
        print("   vi   prun: ", int(prune_mask_vi.sum()))
        prune_mask = torch.logical_or(prune_mask, prune_mask_vi)
        print("logic or: ", int(prune_mask.sum()))
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N]
        print("logic and anchor mask, purn: ", int(prune_mask.sum()))
        print("scale prun: ", int(prune_mask_big_scale.sum()))
        prune_mask_big_scale = (self.get_scaling[:, 0:3].max(dim=1)[0] > max_anchor_size)
        prune_mask = torch.logical_or(prune_mask, prune_mask_big_scale)
        print("final prun: ", int(prune_mask.sum()))
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        dense_score = self.dense_score.view([-1, self.n_offsets])[~prune_mask]
        dense_score = dense_score.view([-1, 1])
        del self.dense_score
        self.dense_score = dense_score.zero_()
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.vi_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum
        
        temp_vi_accum = self.vi_accum[~prune_mask]
        del self.vi_accum
        self.vi_accum = temp_vi_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        voxel_visible_mask = torch.cat((voxel_visible_mask, torch.zeros(self.get_anchor.shape[0]-voxel_visible_mask.shape[0], dtype=torch.bool, device="cuda")))

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        voxel_visible_mask = voxel_visible_mask[~prune_mask]

        return voxel_visible_mask

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.color_dist_dim+self.appearance_dim+self.level_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            vi_mlp = torch.jit.trace(self.mlp_vi, (torch.rand(1, self.feat_dim+self.view_dim+self.vi_dist_dim+self.level_dim).cuda()))
            vi_mlp.save(os.path.join(path, 'vi_mlp.pt'))
            if self.use_feat_bank:
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+self.level_dim).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
            if self.appearance_dim > 0:
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
            self.train()
        elif mode == 'unite':
            param_dict = {}
            param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
            param_dict['cov_mlp'] = self.mlp_cov.state_dict()
            param_dict['color_mlp'] = self.mlp_color.state_dict()
            param_dict['vi_mlp'] = self.mlp_vi.state_dict()
            if self.use_feat_bank:
                param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                param_dict['appearance'] = self.embedding_appearance.state_dict()
            torch.save(param_dict, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            self.mlp_vi = torch.jit.load(os.path.join(path, 'vi_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            self.mlp_vi.load_state_dict(checkpoint['vi_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

    def update_anchor(self, visible_mask):
        # anchor = self.get_anchor[visible_mask]
        # mask_valid = (self.get_opacity_weight[visible_mask] <= 0).squeeze()
        grid_offsets = self.get_offset[visible_mask]
        # grid_offsets[mask_valid] = 0.0
        # anchor_valid_num = torch.sum(~mask_valid, dim=1)
        offsets_delta = torch.sum(grid_offsets, dim=1) / self.n_offsets

        anchor_radii = self.get_scaling[visible_mask,:3]

        self._offset[visible_mask] = grid_offsets - offsets_delta.unsqueeze(dim=1)
        self._anchor[visible_mask] += offsets_delta * anchor_radii
        # self._offset[visible_mask] = self._offset[visible_mask].clamp(min=-1.0, max=1.0)

    def reduce_anchor_size(self):
        anchor_size_new = self.scaling_inverse_activation(self.get_scaling * 0.9)
        optimizable_tensors = self.replace_tensor_to_optimizer(anchor_size_new, "anchor_size")
        self._anchor_size = optimizable_tensors["anchor_size"]
