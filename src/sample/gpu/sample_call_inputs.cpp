#include <iomanip>
#include <iostream>

#include <cstdint>
#include <cstdlib>

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

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_sample_model(MemRefDescriptor<float, 2> *output,
                               MemRefDescriptor<float, 2> *lhs,
                               MemRefDescriptor<float, 2> *rhs);
}

int main() {
  float *lhsData = nullptr;
  float *rhsData = nullptr;
  cudaMallocManaged(&lhsData, 3 * 4 * sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged(&rhsData, 4 * 5 * sizeof(float), cudaMemAttachGlobal);

  for (int i = 0; i < 3 * 4; ++i)
    lhsData[i] = 1.0f;
  for (int i = 0; i < 4 * 5; ++i)
    rhsData[i] = 1.0f;

  MemRefDescriptor<float, 2> lhs = {
      lhsData, lhsData, 0, {3, 4}, {4, 1}};
  MemRefDescriptor<float, 2> rhs = {
      rhsData, rhsData, 0, {4, 5}, {5, 1}};
  MemRefDescriptor<float, 2> output = {};

  _mlir_ciface_sample_model(&output, &lhs, &rhs);
  cudaDeviceSynchronize();

  for (int64_t i = 0; i < output.sizes[0]; ++i) {
    for (int64_t j = 0; j < output.sizes[1]; ++j)
      std::cout << std::fixed << std::setprecision(5)
                << output.aligned[i * output.strides[0] + j] << ' ';
    std::cout << "\n";
  }

  cudaFree(output.allocated);
  cudaFree(lhsData);
  cudaFree(rhsData);
  return 0;
}
