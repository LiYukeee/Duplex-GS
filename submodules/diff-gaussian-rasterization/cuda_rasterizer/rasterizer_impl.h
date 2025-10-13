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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		// size_t scan_size;
		float* depths;
		// char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		// uint32_t* point_offsets;
		// uint32_t* tiles_touched;
		bool* valid_gs;
		float* vi;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		float* weight_sum;

		float* accum_opacity;
		float* final_anchor_correct;
		uint32_t* n_contrib;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct GS_BinningState
	{
		// size_t sorting_size;
		// uint64_t* point_list_keys_unsorted;
		// uint64_t* point_list_keys;
		// uint32_t* point_list_unsorted;
		uint32_t* point_list;
		// char* list_sorting_space;

		static GS_BinningState fromChunk(char*& chunk, size_t P);
	};

	struct AnchorState  // record the state of the anchor
	{
		// used for inclusive sum
		size_t scan_size;
		char *scanning_space;
		// anchor footprint
		// int *radii;
		float2 *means2D; // anchor_xy_image
		uint32_t *tiles_touched; // touched anchors for each tile
		float *depths;
		uint32_t *anchor_offsets;
		uint32_t *anchor_num_valid_gs;

		static AnchorState fromChunk(char *&chunk, size_t P);
	};

	struct AC_BinningState
	{
		// for RadixSort
		size_t sorting_size;
		char* list_sorting_space;

		// for InclusiveSum
		size_t scan_size;
		char *scanning_space;

		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		uint32_t* point_list_num_valid_gs;
		uint32_t* point_list_offset;

		static AC_BinningState fromChunk(char*& chunk, size_t P);
	};


	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};