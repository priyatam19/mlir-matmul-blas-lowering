#include "gpu_bmm_benchmark.h"

extern "C" void _mlir_ciface_gpu_bmm_long(BmmMemRefDescriptor<float, 3> *,
                                          BmmMemRefDescriptor<float, 3> *,
                                          BmmMemRefDescriptor<float, 3> *);

int main() {
  return runGpuBmmBenchmark<12, 512, 64, 512>("bmm_long",
                                              _mlir_ciface_gpu_bmm_long);
}
