#include "gpu_conv_benchmark.h"

extern "C" void
_mlir_ciface_gpu_conv_resnet_stem(ConvMemRefDescriptor<float, 4> *,
                                  ConvMemRefDescriptor<float, 4> *,
                                  ConvMemRefDescriptor<float, 4> *);

int main() {
  return runGpuConvBenchmark<1, 3, 230, 230, 64, 7, 7, 112, 112, 2, 2>(
      "conv_resnet_stem", _mlir_ciface_gpu_conv_resnet_stem);
}
