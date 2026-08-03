#include "gpu_conv_benchmark.h"

extern "C" void
_mlir_ciface_gpu_conv_resnet_block(ConvMemRefDescriptor<float, 4> *,
                                   ConvMemRefDescriptor<float, 4> *,
                                   ConvMemRefDescriptor<float, 4> *);

int main() {
  return runGpuConvBenchmark<1, 64, 58, 58, 64, 3, 3, 56, 56>(
      "conv_resnet_block", _mlir_ciface_gpu_conv_resnet_block);
}
