#include "gpu_conv_benchmark.h"

extern "C" void
_mlir_ciface_gpu_conv_pointwise(ConvMemRefDescriptor<float, 4> *,
                                ConvMemRefDescriptor<float, 4> *,
                                ConvMemRefDescriptor<float, 4> *);

int main() {
  return runGpuConvBenchmark<1, 32, 112, 112, 64, 1, 1, 112, 112>(
      "conv_pointwise", _mlir_ciface_gpu_conv_pointwise);
}
