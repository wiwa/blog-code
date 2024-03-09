#pragma once

#include <thrust/random.h>

#include "utils.cuh"

template <class T> struct HArr {
  T *arr;
  u32 size;

  friend auto operator<=>(const HArr<T> &, const HArr<T> &) = default;

  static HArr<T> alloc(u32 size) {
    T *ts = host::malloc<T>(size);
    return HArr<T>{ts, size};
  }

  inline T &operator[](u32 idx) { return arr[idx]; }
};

template <class T> struct DArr {
  T *arr;
  u32 size;

  friend auto operator<=>(const DArr<T> &, const DArr<T> &) = default;

  HArr<T> to_harr(STREAM(s)) {
    T *h_arr = host::malloc<T>(size);
    gpu::d2hCpy(h_arr, arr, sizeof(T) * size, s);
    return HArr{h_arr, size};
  }

  __host__ __device__ inline bool is_empty() const {
    return arr == nullptr || size == 0;
  }

  __host__ __device__ inline T *begin() const { return arr; }

  __host__ __device__ inline T *end() const { return arr + size; }

  __host__ __device__ static constexpr DArr<T> empty() {
    return DArr<T>{nullptr, 0};
  }

  static DArr<T> zeroes_async(u32 size, STREAM(s)) {
    T *d_arr;
    cudaMallocAsync(&d_arr, size * sizeof(T), s);
    cudaMemsetAsync(d_arr, 0, size * sizeof(T), s);
    return DArr<T>{d_arr, size};
  }

  static DArr<T> from_host(T *h_arr, u32 size, STREAM(s)) {
    T *d_arr;
    if (s == 0) {
      cudaMalloc(&d_arr, size * sizeof(T));
      cudaMemcpy(d_arr, h_arr, size * sizeof(T), cudaMemcpyHostToDevice);
    } else {
      cudaMallocAsync(&d_arr, size * sizeof(T), s);
      cudaMemcpyAsync(d_arr, h_arr, size * sizeof(T), cudaMemcpyHostToDevice,
                      s);
    }
    return DArr<T>{d_arr, size};
  }
};

namespace gpu {

__device__ u32 randint(u32 seed, u32 min, u32 max) {
  thrust::default_random_engine rng(seed);
  thrust::uniform_int_distribution<u32> dist(min, max);
  // The first value seems questionably distributed; discard the first two
  u32 first = dist(rng);
  u32 second = dist(rng);
  return second;
}

__global__ void randints(u32 seed, u32 *results, u32 count) {
  unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadid < count) {
    results[threadid] = randint(seed + threadid, 1, UINT32_MAX);
  }
}

u32 *d_randints(const u64 seed, const u32 count) {
  size_t bytes = sizeof(u32) * count;
  size_t num_ints = bytes / sizeof(u32);

  u32 *device_results;
  cudaMalloc(&device_results, bytes);
  cudaMemset(device_results, 0, bytes);

  int mingridsize;
  int threadblocksize;
  cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, randints,
                                     0, 0);

  int gridsize = ((u32)num_ints + threadblocksize - 1) / threadblocksize;
  randints<<<gridsize, threadblocksize>>>(seed, device_results, num_ints);

  cudaDeviceSynchronize();
  gpu::errChk(cudaGetLastError());

  return device_results;
}
} // namespace gpu