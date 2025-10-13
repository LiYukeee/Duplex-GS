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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "debug_utils.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::AnchorState CudaRasterizer::AnchorState::fromChunk(char *&chunk, size_t P)
{
	AnchorState anchors;
	// obtain(chunk, anchors.radii, P, 128);
	obtain(chunk, anchors.means2D, P, 128);
	obtain(chunk, anchors.depths, P, 128);
	obtain(chunk, anchors.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, anchors.scan_size, anchors.tiles_touched, anchors.tiles_touched, P);
	obtain(chunk, anchors.scanning_space, anchors.scan_size, 128);
	obtain(chunk, anchors.anchor_offsets, P, 128);
	obtain(chunk, anchors.anchor_num_valid_gs, P, 128);
	return anchors;
}

CudaRasterizer::AC_BinningState CudaRasterizer::AC_BinningState::fromChunk(char *&chunk, size_t P)
{
	AC_BinningState binning;
	// for sort
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	// for generate Gaussian list
	obtain(chunk, binning.point_list_num_valid_gs, P, 128);
	obtain(chunk, binning.point_list_offset, P, 128);
	// Allocate space for radix sorting
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	// Allocate space for InclusiveSum
	cub::DeviceScan::InclusiveSum(
		nullptr, 
		binning.scan_size, 
		binning.point_list_num_valid_gs, 
		binning.point_list_num_valid_gs, 
		P);
	obtain(chunk, binning.scanning_space, binning.scan_size, 128);
	return binning;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	// obtain(chunk, geom.tiles_touched, P, 128);
	// cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	// obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	// obtain(chunk, geom.point_offsets, P, 128);
	obtain(chunk, geom.valid_gs, P, 128);
	obtain(chunk, geom.vi, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.weight_sum, N, 128);

	obtain(chunk, img.accum_opacity, N, 128);
	obtain(chunk, img.final_anchor_correct, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	return img;
}

CudaRasterizer::GS_BinningState CudaRasterizer::GS_BinningState::fromChunk(char*& chunk, size_t P)
{
	GS_BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	// obtain(chunk, binning.point_list_unsorted, P, 128);
	// obtain(chunk, binning.point_list_keys, P, 128);
	// obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	// cub::DeviceRadixSort::SortPairs(
	// 	nullptr, binning.sorting_size,
	// 	binning.point_list_keys_unsorted, binning.point_list_keys,
	// 	binning.point_list_unsorted, binning.point_list, P);
	// obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> anchorBuffer,
	std::function<char* (size_t)> anchor_binningBuffer,
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> gs_binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const float sigma,
	const float weight_background,
	const int P_anchor, int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* anchor_means3D,
	const float* anchor_scales,
	const float* anchor_rotations,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* vi,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	// int* num_all_gs,
	// int* num_val_gs,
	// int* num_render_gs,
	int* radii,
	bool debug,
	const float ET_grade,
	const bool if_depth_correct
	// const bool if_get_gs_skip_ratio
	)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);


	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Define some variables in advance.
	GS_BinningState gs_binningState;
	int num_rendered_gs;

	//////////////////////////////////////////////////////////
	/////////////////// anchor search ///////////////////
	size_t anchor_chunk_size = required<AnchorState>(P_anchor);
	char *anchor_chunkptr = anchorBuffer(anchor_chunk_size);
	AnchorState anchorState = AnchorState::fromChunk(anchor_chunkptr, P_anchor);

	CHECK_CUDA(FORWARD::preprocess_gaussian(
		if_depth_correct,
		sigma,
		// GS parameters
		P,
		means3D,
		(glm::vec3 *)scales,
		scale_modifier,
		(glm::vec4 *)rotations,
		opacities,
		vi,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3 *)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		geomState.means2D,
		geomState.depths,
		geomState.internal_radii,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.valid_gs,
		tile_grid,
		prefiltered
	),debug)

	// preprocess anchor generate num_valid_gaussian
	CHECK_CUDA(FORWARD::preprocess_anchor(
		// anchor parameters
		P_anchor,
		anchor_means3D,
		(glm::vec3 *)anchor_scales,
		(glm::vec4 *)anchor_rotations,
		radii,
		anchorState.means2D,
		anchorState.tiles_touched,
		anchorState.anchor_num_valid_gs,
		anchorState.depths,
		// GS parameters
		geomState.valid_gs,
		// others
		viewmatrix, projmatrix,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		tile_grid,
		prefiltered),
	debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(
		cub::DeviceScan::InclusiveSum(
			anchorState.scanning_space,
			anchorState.scan_size,
			anchorState.tiles_touched,
			anchorState.anchor_offsets,
			P_anchor
		),debug
	)

	// Retrieve total number of Anchor instances to launch and resize aux buffers
	int num_rendered_ac;
	CHECK_CUDA(cudaMemcpy(&num_rendered_ac, anchorState.anchor_offsets + P_anchor - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	// For each instance to be rendered, produce adequate [ tile ] key
	// and corresponding dublicated Anchor indices to be sorted
	AC_BinningState ac_binningState;
	size_t binning_chunk_size_ac = required<AC_BinningState>(num_rendered_ac);
	char* binning_chunkptr_ac = anchor_binningBuffer(binning_chunk_size_ac);
	ac_binningState = AC_BinningState::fromChunk(binning_chunkptr_ac, num_rendered_ac);

	duplicateWithKeys_anchor<<<(P_anchor + 255) / 256, 256>>>(
		P_anchor,
		anchorState.depths,
		anchorState.means2D,
		anchorState.anchor_offsets,
		radii,
		ac_binningState.point_list_keys_unsorted,
		ac_binningState.point_list_unsorted,
		tile_grid
	) CHECK_CUDA(, debug);

	// Sort complete list of (duplicated) Anchor indices by keys
	// [ anchor's tile -- anchor ID -- num of valid gs ]
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		ac_binningState.list_sorting_space,
		ac_binningState.sorting_size,
		ac_binningState.point_list_keys_unsorted, ac_binningState.point_list_keys,
		ac_binningState.point_list_unsorted, ac_binningState.point_list,
		num_rendered_ac, 0, 32 + bit), debug)
	// float f_sorting_szie = float(ac_binningState.sorting_size);
	// float f_ac_list_size = num_rendered_ac * 2;
	// printf("num_render_ac: %d, sort_size: %f(MB), times: %f\n", num_rendered_ac, f_sorting_szie / 1024.0f / 1024.0f, f_sorting_szie / f_ac_list_size);

	sort_num_valid_gs<<<(num_rendered_ac + 255) / 256, 256>>>(
		num_rendered_ac,
		ac_binningState.point_list,
		anchorState.anchor_num_valid_gs,
		ac_binningState.point_list_num_valid_gs
	) CHECK_CUDA(, debug)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		ac_binningState.scanning_space,
		ac_binningState.scan_size,
		ac_binningState.point_list_num_valid_gs,
		ac_binningState.point_list_offset, 
		num_rendered_ac
		),debug)

	// Generate a list of Gaussian that need to be rendered
	// int num_rendered_gs;
	CHECK_CUDA(cudaMemcpy(&num_rendered_gs, ac_binningState.point_list_offset + num_rendered_ac - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);

	// GS_BinningState gs_binningState;
	size_t binning_chunk_size_gs = required<GS_BinningState>(num_rendered_gs);
	char* binning_chunkptr_gs = gs_binningBuffer(binning_chunk_size_gs);
	gs_binningState = GS_BinningState::fromChunk(binning_chunkptr_gs, num_rendered_gs);

	generate_render_gs_list<<<(num_rendered_ac + 255) / 256, 256>>>(
		num_rendered_ac,
		ac_binningState.point_list,
		ac_binningState.point_list_offset,
		geomState.valid_gs,
		gs_binningState.point_list
	) CHECK_CUDA(, debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered_gs > 0)
	{
		identifyTileRanges_anchor <<< (num_rendered_ac + 255) / 256, 256 >>> (
			num_rendered_ac,
			ac_binningState.point_list_keys,
			ac_binningState.point_list_offset,
			imgState.ranges
		) CHECK_CUDA(, debug)
	}


	/////////////////// end anchor search ///////////////////
	//////////////////////////////////////////////////////////
	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	CHECK_CUDA(FORWARD::render(
		// if_get_gs_skip_ratio,
		// num_all_gs,
		// num_val_gs,
		// num_render_gs,
		ET_grade,
		weight_background,
		tile_grid, block,
		imgState.ranges,
		gs_binningState.point_list,
		width, height,
		geomState.means2D,
		geomState.internal_radii,
		feature_ptr,
		geomState.conic_opacity,
		imgState.weight_sum,
		imgState.accum_opacity,
		imgState.final_anchor_correct,
		imgState.n_contrib,
		background,
		out_color
		), debug)

	return num_rendered_gs;
}


void CudaRasterizer::Rasterizer::visible_filter(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int M,
	const int width, int height,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	// dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::filter_preprocess(
		P, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		cov3D_precomp,
		viewmatrix, projmatrix,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.cov3D,
		tile_grid,
		prefiltered
	), debug)

}


// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	float* dense_score, //add
	const float sigma,  //add
	const float* render_image,  //add
	const float* opacities,
	const float* vi,
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dvi,  //add
	float* dL_ddepth,  //add
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dsigma,  //add
	float* dL_dweight_background,  //add
	bool debug,
	const float ET_grade,
	const bool if_depth_correct
	)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	GS_BinningState binningState = GS_BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		dense_score,
		ET_grade,
		render_image,
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.internal_radii,
		geomState.conic_opacity,
		color_ptr,
		imgState.weight_sum,
		imgState.accum_opacity,
		imgState.final_anchor_correct,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dweight_background,
		dL_dvi
		), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(
		if_depth_correct,
		sigma,
		geomState.valid_gs,
		geomState.depths,
		opacities,
		vi,
		dL_dopacity,
		dL_dvi,
		dL_ddepth,
		dL_dsigma,
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)

}