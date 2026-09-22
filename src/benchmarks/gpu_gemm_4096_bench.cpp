#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_4096(MemRefDescriptor<float, 2> *,
                                             MemRefDescriptor<float, 2> *,
                                             MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<4096, 4096, 4096>("gemm_4096",
                                               _mlir_ciface_gpu_gemm_4096);
}
