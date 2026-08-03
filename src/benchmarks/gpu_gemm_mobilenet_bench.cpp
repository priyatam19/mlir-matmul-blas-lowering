#include "gpu_gemm_benchmark.h"

extern "C" void _mlir_ciface_gpu_gemm_mobilenet(MemRefDescriptor<float, 2> *,
                                                MemRefDescriptor<float, 2> *,
                                                MemRefDescriptor<float, 2> *);

int main() {
  return runGpuGemmBenchmark<3136, 32, 64>("gemm_mobilenet",
                                           _mlir_ciface_gpu_gemm_mobilenet);
}
