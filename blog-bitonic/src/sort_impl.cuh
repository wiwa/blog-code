#pragma once

#include "utils.cuh"

// sort 32 T's with warp primitives
// Implementing the non-alternating version found here:
// https://www.wikiwand.com/en/Bitonic_sorter
template <class T>
__device__ static inline void warp_sort_bitonic(u32 threadid, T *data) {
  u32 lane = gpu::get_lane(threadid);
  u32 other_lane;

  u32 width; // 'vertical' width of sort network block
  u32 subdepth;

  u32 datum;
  u32 other_datum;

  // load data into register
  datum = data[lane];

  for (width = 2; width <= 32; width = width << 1) {
    // first block of symmetric bitonic merge
    other_lane = lane ^ (width - 1);
    // swap if necessary between lane and other_lane
    //  such that smaller datum is in smaller lane index
    other_datum = __shfl_sync(gpu::all_lanes, datum, other_lane);
    if (lane < other_lane) {
      datum = min(datum, other_datum);
    } else {
      datum = max(datum, other_datum);
    }
    // rest of blocks are the same except the 'other_lane' calculation
    for (subdepth = width >> 2; subdepth > 0; subdepth = subdepth >> 1) {
      other_lane = lane ^ subdepth;
      other_datum = __shfl_sync(gpu::all_lanes, datum, other_lane);
      if (lane < other_lane) {
        datum = min(datum, other_datum);
      } else {
        datum = max(datum, other_datum);
      }
    }
  }

  data[lane] = datum;
  __syncwarp();
}

// same as warp_sort_bitonic but with shared memory instead of __shlf_sync
template <class T>
__device__ static inline void warp_sort_bitonic_smem(u32 threadid, T *data, u32* smem) {
  u32 lane = threadIdx.x;
  u32 other_lane;

  u32 width;
  u32 subdepth;

  u32 datum;
  u32 other_datum;

  datum = data[lane];
  __syncwarp();

  for (width = 2; width <= 32; width = width << 1) {
    other_lane = lane ^ (width - 1);
    smem[lane] = datum;
    __syncwarp();
    other_datum = smem[other_lane];
    if (lane < other_lane) {
      datum = min(datum, other_datum);
    } else {
      datum = max(datum, other_datum);
    }
    smem[lane] = datum;
    for (subdepth = width >> 2; subdepth > 0; subdepth = subdepth >> 1) {
      other_lane = lane ^ subdepth;
      __syncwarp();
      other_datum = smem[other_lane];
      if (lane < other_lane) {
        datum = min(datum, other_datum);
      } else {
        datum = max(datum, other_datum);
      }
      smem[lane] = datum;
    }
  }

  data[lane] = datum;
  __syncwarp();
}

template <class T> __global__ void test_sort_bitonic(T *data, u32 n_data) {
  const u32 threadid = gpu::threadid();
  const u32 stride = gpu::stride();
  const u32 end = n_data;
  // every lane in a warp should have the same index (start of warp)
  const u32 index = gpu::get_warp(threadid) * gpu::warp_sz;

  for (u32 i = index; i < end; i += stride) {
    warp_sort_bitonic(threadid, data + i);
  }
}

template <class T> __global__ void test_sort_bitonic_smem(T *data, u32 n_data) {
  const u32 threadid = gpu::threadid();
  const u32 stride = gpu::stride();
  const u32 end = n_data;
  const u32 block_sz = blockDim.x;
  // every lane in a block should have the same index (start of block)
  const u32 index = blockIdx.x * block_sz;

  // 2048 is max block size but it fails(???); just use 1024
  // We should only bench on single-warp blocks anyway (32 threads)
  // There seems to be no perf diff in allocating for 32 or 1024
  __shared__ u32 smem[1024];

  // if we use "data + i" instead of smem (i.e. directly use global mem),
  // it's about 10% slower. 
  for (u32 i = index; i < end; i += stride) {
    warp_sort_bitonic_smem(threadid, data + i, smem);
  }
}

template <class T>
__device__ static inline bool is_sorted_window(u32 threadid, T *data) {
  u32 lane = gpu::get_lane(threadid);
  if (lane < gpu::warp_mask) {
    u32 left = data[lane];
    u32 right = data[lane + 1];
    if (left > right) {
      // printf("is_sorted_window failed (<): %d %d %d\n", lane, data[lane],
      //        data[lane + 1]);
      return false;
    }
  }
  return true;
}

__global__ void check_sorted_windows(u32 *data, u32 n_data, u32 *n_failures) {
  const u32 threadid = gpu::threadid();
  const u32 stride = gpu::stride();
  const u32 end = n_data;
  const u32 index = gpu::get_warp(threadid) * gpu::warp_sz;

  for (u32 i = index; i < end; i += stride) {
    bool is_sorted = is_sorted_window(threadid, data + i);
    if (!is_sorted) {
      atomicAdd(n_failures, 1);
    }
  }
}
