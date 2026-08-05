#include "gpu_bmm_benchmark.h"

extern "C" void _mlir_ciface_gpu_bmm_irregular(BmmMemRefDescriptor<float, 3> *,
                                               BmmMemRefDescriptor<float, 3> *,
                                               BmmMemRefDescriptor<float, 3> *);

int main() {
  return runGpuBmmBenchmark<7, 129, 65, 127>("bmm_irregular",
                                             _mlir_ciface_gpu_bmm_irregular);
}
