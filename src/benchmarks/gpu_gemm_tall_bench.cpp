#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_tall(MemRefDescriptor<float, 2> *,
                                             MemRefDescriptor<float, 2> *,
                                             MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<4096, 1024, 256>("gemm_tall",
                                              _mlir_ciface_gpu_gemm_tall);
}
