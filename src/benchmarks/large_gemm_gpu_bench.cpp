#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_512(MemRefDescriptor<float, 2> *,
                                          MemRefDescriptor<float, 2> *,
                                          MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<512, 256, 512>("gemm_512",
                                            _mlir_ciface_gpu_gemm_512);
}
