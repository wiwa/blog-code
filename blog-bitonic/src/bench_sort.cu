#include <cxxopts.hpp>

#include "data.cuh"
#include "sort_impl.cuh"
#include "utils.cuh"

void time_bitonic(u32 data_pow, u32 grid_sz, u32 block_sz, bool smem = false) {
  u32 n_ints = 1 << data_pow;
  u32 n_windows = n_ints / gpu::warp_sz;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  u32 *d_ints = gpu::d_randints(0, n_ints);

  // ensure that we can actually check the results
  DArr<u32> d_failures = DArr<u32>::zeroes_async(1, stream);
  HArr<u32> h_failures = d_failures.to_harr(stream);

  check_sorted_windows<<<grid_sz, block_sz, 0, stream>>>(d_ints, n_ints,
                                                         d_failures.arr);
  gpu::d2hCpy(h_failures.arr, d_failures.arr, 1, stream);
  gpu::streamSyncChk(stream);

  if (h_failures[0] == n_windows) {
    printf("Initial data has all sorted windows??\n");
    exit(1);
  }

  // warmup
  for (u32 i = 0; i < 3; i++) {
    gpu::emptyKernel<<<768, 768, 0, stream>>>();
    gpu::streamSyncChk(stream);
  }

  u32 timed = U32_MAX;
  if (smem) {
    timed = gpu::timeFnMicros([&]() {
      test_sort_bitonic_smem<u32>
          <<<grid_sz, block_sz, 0, stream>>>(d_ints, n_ints);
      cudaStreamSynchronize(stream);
    });
  } else {
    timed = gpu::timeFnMicros([&]() {
      test_sort_bitonic<u32><<<grid_sz, block_sz, 0, stream>>>(d_ints, n_ints);
      cudaStreamSynchronize(stream);
    });
  }
  gpu::errChk(cudaGetLastError());

  printf("Bitonic sort <<<%u, %u>>> (smem=%u) took %llu us\n", grid_sz,
         block_sz, smem, timed);

  cudaMemsetAsync(d_failures.arr, 0, sizeof(u32), stream);
  check_sorted_windows<<<grid_sz, block_sz, 0, stream>>>(d_ints, n_ints,
                                                         d_failures.arr);
  gpu::d2hCpy(h_failures.arr, d_failures.arr, 1, stream);
  gpu::streamSyncChk(stream);

  assert(h_failures[0] == 0);
  printf("Data is in sorted windows.\n");
}

int main(int argc, char **argv) {

  cxxopts::Options options("BenchSort", "Benching some CUDA stuff");

  options.add_options()(
      "data", "the number of integers to process as a power of 2 (20 for 2^20)",
      cxxopts::value<u32>())("grid", "size of the CUDA grid",
                             cxxopts::value<u32>())(
      "block", "size of the CUDA block", cxxopts::value<u32>()->default_value("32"));
  options.parse_positional({"data", "grid", "block"});

  auto params = options.parse(argc, argv);

  for (std::string s : params.unmatched()) {
    printf("Unmatched argument: %s\n", s.c_str());
  }

  u32 data = params["data"].as<u32>();
  u32 grid = params["grid"].as<u32>();
  u32 block = params["block"].as<u32>();
  printf("Arguments:\nData: 2^%u, Grid: %u, Block: %u\n", data, grid, block);

  gpu::emptyKernel<<<1, 1>>>();
  cudaDeviceSynchronize();

  time_bitonic(data, grid, block, false); // in-register sort
  time_bitonic(data, grid, block, true);  // shared mem sort

  return 0;
}