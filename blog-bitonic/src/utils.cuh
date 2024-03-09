#pragma once

#include "stdint.h"
#include "stdio.h"
#include <assert.h>
#include <chrono>
#include <cuda/stream_ref>

#define STREAM(s) cudaStream_t s = 0

typedef uint32_t u32;
typedef unsigned long long int u64;

constexpr u32 U32_MAX = ((u32)-1);
constexpr u64 U64_MAX = ((u64)-1);

namespace gpu {

__global__ void emptyKernel(void){};

#define errChk(code) gpuAssert(code, __FILE__, __LINE__)
__host__ inline void gpuAssert(cudaError_t code, const char *file, int line,
                               bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) {
      fprintf(stderr, "Aborting due to GPUAssert\n");
      exit(code);
    }
  }
}

#define errChkDevice(code) gpuAssertDevice(code, __FILE__, __LINE__)
__device__ inline void gpuAssertDevice(cudaError_t code, const char *file,
                                       int line) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  }
}

static void streamSyncChk(cudaStream_t stream = cudaStreamDefault) {
  if (stream == cudaStreamDefault) {
    cudaDeviceSynchronize();
  } else {
    cudaStreamSynchronize(stream);
  }
  gpu::errChk(cudaGetLastError());
}

constexpr u32 warp_sz = 32;
constexpr u32 warp_pow = 5;
constexpr u32 warp_mask = warp_sz - 1;
constexpr u32 all_lanes = 0xFFFFFFFF;

__device__ inline u32 threadid() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ inline u32 stride() { return gridDim.x * blockDim.x; }
__device__ inline u32 warp_stride() { return stride() >> warp_pow; }
__device__ inline u32 get_warp(u32 threadid) { return threadid >> warp_pow; }
__device__ inline u32 get_lane(u32 threadid) { return threadid & warp_mask; }

inline void streamMemcpy(void *dst, void *src, u32 nbytes,
                         cudaMemcpyKind direction, cudaStream_t stream) {
  if (stream == 0) {
    cudaMemcpy(dst, src, nbytes, direction);
  } else {
    cudaMemcpyAsync(dst, src, nbytes, direction, stream);
  }
}

inline void d2hCpy(void *h_ptr, void *d_ptr, u32 nbytes,
                   cudaStream_t stream = 0) {
  streamMemcpy(h_ptr, d_ptr, nbytes, cudaMemcpyDeviceToHost, stream);
}
inline void h2dCpy(void *d_ptr, void *h_ptr, const u32 nbytes,
                   cudaStream_t stream = 0) {
  streamMemcpy(d_ptr, h_ptr, nbytes, cudaMemcpyHostToDevice, stream);
}

template <typename F> int64_t timeFnMicros(F &&lambda) {
  auto t0 = std::chrono::high_resolution_clock::now();

  lambda();

  auto t1 = std::chrono::high_resolution_clock::now();

  auto d = t1 - t0;
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(d);

  return us.count();
}
} // namespace gpu

namespace host {

template <class T> T *malloc(u32 n) { return (T *)std::malloc(sizeof(T) * n); }

} // namespace host