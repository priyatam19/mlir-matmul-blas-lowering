#include "gpu_conv_benchmark.h"

extern "C" void
_mlir_ciface_gpu_conv_irregular(ConvMemRefDescriptor<float, 4> *,
                                ConvMemRefDescriptor<float, 4> *,
                                ConvMemRefDescriptor<float, 4> *);

int main() {
  return runGpuConvBenchmark<2, 7, 35, 37, 13, 3, 5, 16, 33, 2, 1, 2, 1>(
      "conv_irregular", _mlir_ciface_gpu_conv_irregular);
}
