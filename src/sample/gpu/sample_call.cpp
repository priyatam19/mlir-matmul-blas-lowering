#include <iomanip>
#include <iostream>

#include <cstdint>
#include <cstdio>
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
                               MemRefDescriptor<float, 2> *input);
}

int main(int argc, char *argv[]) {
  float *inputData = nullptr;
  float *outputData = nullptr;

  cudaMallocManaged(&inputData, 3 * 4 * sizeof(float), cudaMemAttachGlobal);
  cudaMallocManaged(&outputData, 3 * 5 * sizeof(float), cudaMemAttachGlobal);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      inputData[i * 4 + j] = 1.0;
    }
  }

  // Create MemRef descriptors
  int64_t offset = 0;
  int64_t input_sizes[2] = {3, 4};
  int64_t input_strides[2] = {4, 1}; // row-major layout

  int64_t output_size[2] = {3, 5};
  int64_t output_strides[2] = {5, 1}; // row-major layout

  MemRefDescriptor<float, 2> inputMemRef = {
      inputData,
      inputData,
      offset,
      {input_sizes[0], input_sizes[1]},
      {input_strides[0], input_strides[1]}};

  MemRefDescriptor<float, 2> outputMemRef = {
      outputData,
      outputData,
      offset,
      {output_size[0], output_size[1]},
      {output_strides[0], output_strides[1]}};

  // Call the model
  _mlir_ciface_sample_model(&outputMemRef, &inputMemRef);
  cudaDeviceSynchronize();

  float *output = (float *)outputMemRef.aligned;

  for (int64_t i = 0; i < output_size[0]; ++i) {
    for (int64_t j = 0; j < output_size[1]; ++j)
      std::cout << std::fixed << std::setprecision(5)
                << output[i * output_strides[0] + j] << ' ';
    std::cout << "\n";
  }

  cudaFree(inputData);
  cudaFree(outputData);

  return 0;
}
