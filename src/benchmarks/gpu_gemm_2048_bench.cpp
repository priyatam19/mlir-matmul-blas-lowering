#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_2048(MemRefDescriptor<float, 2> *,
                                           MemRefDescriptor<float, 2> *,
                                           MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<2048, 2048, 2048>("gemm_2048",
                                               _mlir_ciface_gpu_gemm_2048);
}
