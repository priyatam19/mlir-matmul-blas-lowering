#include "gpu_conv_benchmark.h"

extern "C" void _mlir_ciface_gpu_conv_small(ConvMemRefDescriptor<float, 4> *,
                                            ConvMemRefDescriptor<float, 4> *,
                                            ConvMemRefDescriptor<float, 4> *);

int main() {
  return runGpuConvBenchmark<4, 1, 30, 30, 32, 3, 3, 28, 28>(
      "conv_small", _mlir_ciface_gpu_conv_small);
}
