#include "gpu_bmm_benchmark.h"

extern "C" void _mlir_ciface_gpu_bmm_value(BmmMemRefDescriptor<float, 3> *,
                                           BmmMemRefDescriptor<float, 3> *,
                                           BmmMemRefDescriptor<float, 3> *);

int main() {
  return runGpuBmmBenchmark<32, 128, 64, 64>("bmm_value",
                                             _mlir_ciface_gpu_bmm_value);
}
