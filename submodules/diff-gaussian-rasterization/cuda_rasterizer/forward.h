/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Main rasterization method.
	void render(
		// const bool if_get_gs_skip_ratio,
		// int* num_of_all_gs,
		// int* num_of_valid_gs,
		// int* num_of_render_gs,
		const float ET_grade,
		const float weight_background,
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const int* gs_radii,
		const float* features,
		const float4* conic_opacity,
		float* weight_sum,
		float* accum_opacity,
		float* final_anchor_correct,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color
		);


	void filter_preprocess(int P, int M,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float* cov3Ds,
		const dim3 grid,
		bool prefiltered);

	void preprocess_gaussian(
		const bool if_depth_correct,
		const float sigma,
		int P,
		const float *means3D,
		const glm::vec3 *scales,
		const float scale_modifier,
		const glm::vec4 *rotations,
		const float *opacities,
		const float *vi,
		const float *cov3D_precomp,
		const float *colors_precomp,
		const float *viewmatrix,
		const float *projmatrix,
		const glm::vec3 *cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		float2 *points_xy_image,
		float *depths,
		int *gs_radii,
		float *cov3Ds,
		float *colors,
		float4 *conic_opacity,
		bool *valid_gs,
		const dim3 grid,
		bool prefiltered
	);

	void preprocess_anchor(
		int P_anchor,
		const float *anchor_means3D,
		const glm::vec3* anchor_scales,
		const glm::vec4* anchor_rotations,
		int *anchor_radii,
		float2 *anchor_means2D,
		uint32_t *anchor_tiles_touched,
		uint32_t *anchor_num_valid_gs,
		float *anchor_depths,
		bool *valid_gs,
		const float *viewmatrix,
		const float *projmatrix,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const dim3 grid,
		bool prefiltered
	);

}

__global__ void duplicateWithKeys_anchor(
	int P,
	const float *anchor_depths,
	const float2 *points_anchor,
	const uint32_t *offsets,
	int *radii,
	uint64_t *anchor_keys_unsorted,
	uint32_t *anchor_values_unsorted,
	dim3 grid
);

__global__ void sort_num_valid_gs(
	int num_rendered_ac,
	const uint32_t *sort_index,
	const uint32_t *anchor_valid_gs_unsorted,
	uint32_t *anchor_valid_gs_sorted
);

__global__ void generate_render_gs_list(
	const uint32_t num_render_ac,
	const uint32_t* __restrict__ anchor_list,
	const uint32_t* __restrict__ offset,
	const bool* __restrict__ valid_gs,
	uint32_t *point_list
);

__global__ void identifyTileRanges_anchor(
	int L,
	uint64_t* point_list_keys,
	uint32_t* point_list_offset,
	uint2* ranges
);


#endif