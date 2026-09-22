#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_wide(MemRefDescriptor<float, 2> *,
                                             MemRefDescriptor<float, 2> *,
                                             MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<256, 1024, 4096>("gemm_wide",
                                              _mlir_ciface_gpu_gemm_wide);
}
