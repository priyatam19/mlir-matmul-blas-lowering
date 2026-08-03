#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_1024(MemRefDescriptor<float, 2> *,
                                           MemRefDescriptor<float, 2> *,
                                           MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<1024, 1024, 1024>("gemm_1024",
                                               _mlir_ciface_gpu_gemm_1024);
}
