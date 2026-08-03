#include "gpu_bmm_benchmark.h"

extern "C" void _mlir_ciface_gpu_bmm_bert(BmmMemRefDescriptor<float, 3> *,
                                          BmmMemRefDescriptor<float, 3> *,
                                          BmmMemRefDescriptor<float, 3> *);

int main() {
  return runGpuBmmBenchmark<12, 128, 64, 128>("bmm_bert",
                                              _mlir_ciface_gpu_bmm_bert);
}
