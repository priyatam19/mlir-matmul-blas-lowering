// GPU benchmark harness: large_gemm (512x256) x (256x512) -> (512x512)
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

extern "C" {
void *__real_malloc(size_t size);
void __real_free(void *ptr);

void *__wrap_malloc(size_t size) {
  void *ptr = nullptr;
  if (cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) != cudaSuccess)
    return __real_malloc(size);
  return ptr;
}

void __wrap_free(void *ptr) {
  if (ptr)
    cudaFree(ptr);
}
}

template <typename T, int NDIM> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[NDIM];
  int64_t strides[NDIM];
};

extern "C" {
void _mlir_ciface_large_gemm(MemRefDescriptor<float, 2> *result,
                             MemRefDescriptor<float, 2> *A,
                             MemRefDescriptor<float, 2> *B);
}

static MemRefDescriptor<float, 2> make2D(float *data, int64_t rows,
                                         int64_t cols) {
  return {data, data, 0, {rows, cols}, {cols, 1}};
}

int main() {
  float *aData = nullptr;
  float *bData = nullptr;
  cudaMallocManaged(&aData, 512 * 256 * sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged(&bData, 256 * 512 * sizeof(float), cudaMemAttachGlobal);

  for (int i = 0; i < 512 * 256; ++i)
    aData[i] = 0.001f * static_cast<float>(i % 100 + 1);
  for (int i = 0; i < 256 * 512; ++i)
    bData[i] = 0.001f * static_cast<float>((i + 3) % 100 + 1);

  auto a = make2D(aData, 512, 256);
  auto b = make2D(bData, 256, 512);
  MemRefDescriptor<float, 2> c = {};

  constexpr int warmups = 3;
  constexpr int runs = 10;

  for (int i = 0; i < warmups; ++i) {
    _mlir_ciface_large_gemm(&c, &a, &b);
    cudaDeviceSynchronize();
    cudaFree(c.allocated);
    c = {};
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; ++i) {
    _mlir_ciface_large_gemm(&c, &a, &b);
    cudaDeviceSynchronize();
    cudaFree(c.allocated);
    c = {};
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "large_gemm GPU avg inference time: " << std::fixed
            << std::setprecision(9) << (elapsed.count() / runs) << " sec\n";

  _mlir_ciface_large_gemm(&c, &a, &b);
  cudaDeviceSynchronize();
  std::cout << "checksum sample: " << std::fixed << std::setprecision(6)
            << c.aligned[0] << " " << c.aligned[512 * 256 + 256] << " "
            << c.aligned[512 * 512 - 1] << "\n";
  cudaFree(c.allocated);
  cudaFree(aData);
  cudaFree(bData);
  return 0;
}
