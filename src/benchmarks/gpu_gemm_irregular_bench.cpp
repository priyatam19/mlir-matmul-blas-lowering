#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_irregular(MemRefDescriptor<float, 2> *,
                                                MemRefDescriptor<float, 2> *,
                                                MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<513, 257, 509>("gemm_irregular",
                                            _mlir_ciface_gpu_gemm_irregular);
}
