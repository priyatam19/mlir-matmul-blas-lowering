#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

extern "C" {
void *__real_malloc(size_t size);
void __real_free(void *pointer);

void *__wrap_malloc(size_t size) {
  void *pointer = nullptr;
  if (cudaMallocManaged(&pointer, size, cudaMemAttachGlobal) != cudaSuccess)
    return __real_malloc(size);
  return pointer;
}

void __wrap_free(void *pointer) {
  if (pointer)
    cudaFree(pointer);
}
}

template <typename T, int Rank> struct ResidualMemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

extern "C" void
_mlir_ciface_residual_conv_block(ResidualMemRefDescriptor<float, 4> *output,
                                 ResidualMemRefDescriptor<float, 4> *value,
                                 ResidualMemRefDescriptor<float, 4> *weight1,
                                 ResidualMemRefDescriptor<float, 1> *bias1,
                                 ResidualMemRefDescriptor<float, 4> *weight2,
                                 ResidualMemRefDescriptor<float, 1> *bias2);

namespace {

constexpr int Channels = 16;
constexpr int Height = 32;
constexpr int Width = 32;
constexpr size_t ActivationElements =
    static_cast<size_t>(Channels) * Height * Width;
constexpr size_t WeightElements =
    static_cast<size_t>(Channels) * Channels * 3 * 3;

void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

int envInt(const char *name, int defaultValue) {
  const char *value = std::getenv(name);
  if (!value)
    return defaultValue;
  int parsed = std::atoi(value);
  return parsed > 0 ? parsed : defaultValue;
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  return values[static_cast<size_t>((values.size() - 1) * fraction)];
}

size_t activationIndex(int channel, int row, int column) {
  return (static_cast<size_t>(channel) * Height + row) * Width + column;
}

size_t weightIndex(int outputChannel, int inputChannel, int kernelRow,
                   int kernelColumn) {
  return ((static_cast<size_t>(outputChannel) * Channels + inputChannel) * 3 +
          kernelRow) *
             3 +
         kernelColumn;
}

std::vector<float> referenceResidualBlock(const float *value,
                                          const float *weight1,
                                          const float *bias1,
                                          const float *weight2,
                                          const float *bias2) {
  std::vector<float> hidden(ActivationElements);
  std::vector<float> output(ActivationElements);
  auto convolution = [&](const float *input, const float *weight,
                         const float *bias, float *destination,
                         bool addResidual) {
    for (int outputChannel = 0; outputChannel < Channels; ++outputChannel) {
      for (int row = 0; row < Height; ++row) {
        for (int column = 0; column < Width; ++column) {
          float sum = bias[outputChannel];
          for (int inputChannel = 0; inputChannel < Channels; ++inputChannel) {
            for (int kernelRow = 0; kernelRow < 3; ++kernelRow) {
              int inputRow = row + kernelRow - 1;
              if (inputRow < 0 || inputRow >= Height)
                continue;
              for (int kernelColumn = 0; kernelColumn < 3; ++kernelColumn) {
                int inputColumn = column + kernelColumn - 1;
                if (inputColumn < 0 || inputColumn >= Width)
                  continue;
                sum += input[activationIndex(inputChannel, inputRow,
                                             inputColumn)] *
                       weight[weightIndex(outputChannel, inputChannel,
                                          kernelRow, kernelColumn)];
              }
            }
          }
          size_t index = activationIndex(outputChannel, row, column);
          if (addResidual)
            sum += value[index];
          destination[index] = std::max(sum, 0.0f);
        }
      }
    }
  };
  convolution(value, weight1, bias1, hidden.data(), false);
  convolution(hidden.data(), weight2, bias2, output.data(), true);
  return output;
}

std::vector<float> loadPyTorchReference(const char *path) {
  std::vector<float> output(ActivationElements);
  FILE *file = std::fopen(path, "rb");
  if (!file) {
    std::perror("open PyTorch reference");
    std::exit(1);
  }
  size_t elements =
      std::fread(output.data(), sizeof(float), output.size(), file);
  int trailingByte = std::fgetc(file);
  std::fclose(file);
  if (elements != output.size() || trailingByte != EOF) {
    std::fprintf(stderr, "PyTorch reference has an unexpected size: %s\n",
                 path);
    std::exit(1);
  }
  return output;
}

} // namespace

int main() {
  const int warmups = envInt("WARMUPS", 5);
  const int runs = envInt("RUNS", 20);
  const char *lowering = std::getenv("GPU_LOWERING");
  if (!lowering)
    lowering = "block-thread";

  float *valueData = nullptr;
  float *weight1Data = nullptr;
  float *bias1Data = nullptr;
  float *weight2Data = nullptr;
  float *bias2Data = nullptr;
  checkCuda(cudaMallocManaged(&valueData, ActivationElements * sizeof(float)),
            "cudaMallocManaged(value)");
  checkCuda(cudaMallocManaged(&weight1Data, WeightElements * sizeof(float)),
            "cudaMallocManaged(weight1)");
  checkCuda(cudaMallocManaged(&weight2Data, WeightElements * sizeof(float)),
            "cudaMallocManaged(weight2)");
  checkCuda(cudaMallocManaged(&bias1Data, Channels * sizeof(float)),
            "cudaMallocManaged(bias1)");
  checkCuda(cudaMallocManaged(&bias2Data, Channels * sizeof(float)),
            "cudaMallocManaged(bias2)");
  for (size_t i = 0; i < ActivationElements; ++i)
    valueData[i] = 0.002f * static_cast<float>(static_cast<int>(i % 101) - 50);
  for (size_t i = 0; i < WeightElements; ++i) {
    weight1Data[i] =
        0.001f * static_cast<float>(static_cast<int>((i + 11) % 97) - 48);
    weight2Data[i] =
        0.001f * static_cast<float>(static_cast<int>((i + 29) % 89) - 44);
  }
  for (int i = 0; i < Channels; ++i) {
    bias1Data[i] = 0.001f * static_cast<float>(i - 8);
    bias2Data[i] = 0.001f * static_cast<float>(7 - i);
  }

  ResidualMemRefDescriptor<float, 4> value = {
      valueData,
      valueData,
      0,
      {1, Channels, Height, Width},
      {Channels * Height * Width, Height * Width, Width, 1}};
  ResidualMemRefDescriptor<float, 4> weight1 = {weight1Data,
                                                weight1Data,
                                                0,
                                                {Channels, Channels, 3, 3},
                                                {Channels * 9, 9, 3, 1}};
  ResidualMemRefDescriptor<float, 4> weight2 = {weight2Data,
                                                weight2Data,
                                                0,
                                                {Channels, Channels, 3, 3},
                                                {Channels * 9, 9, 3, 1}};
  ResidualMemRefDescriptor<float, 1> bias1 = {
      bias1Data, bias1Data, 0, {Channels}, {1}};
  ResidualMemRefDescriptor<float, 1> bias2 = {
      bias2Data, bias2Data, 0, {Channels}, {1}};
  ResidualMemRefDescriptor<float, 4> output = {};

  int device = 0;
  checkCuda(cudaGetDevice(&device), "cudaGetDevice");
  for (auto [pointer, bytes] :
       {std::pair<void *, size_t>{valueData,
                                  ActivationElements * sizeof(float)},
        {weight1Data, WeightElements * sizeof(float)},
        {weight2Data, WeightElements * sizeof(float)},
        {bias1Data, Channels * sizeof(float)},
        {bias2Data, Channels * sizeof(float)}}) {
    checkCuda(cudaMemPrefetchAsync(pointer, bytes, device), "prefetch input");
  }
  checkCuda(cudaDeviceSynchronize(), "prefetch synchronize");

  _mlir_ciface_residual_conv_block(&output, &value, &weight1, &bias1, &weight2,
                                   &bias2);
  checkCuda(cudaDeviceSynchronize(), "correctness synchronize");
  std::vector<float> reference = referenceResidualBlock(
      valueData, weight1Data, bias1Data, weight2Data, bias2Data);
  if (const char *referencePath = std::getenv("PYTORCH_REFERENCE"))
    reference = loadPyTorchReference(referencePath);
  double maxAbsError = 0.0;
  double maxRelError = 0.0;
  size_t mismatches = 0;
  for (size_t i = 0; i < ActivationElements; ++i) {
    double actual = output.aligned[output.offset + i];
    double expected = reference[i];
    double absoluteError = std::abs(actual - expected);
    double relativeError = absoluteError / std::max(std::abs(expected), 1.0e-6);
    maxAbsError = std::max(maxAbsError, absoluteError);
    maxRelError = std::max(maxRelError, relativeError);
    if (!std::isfinite(actual) ||
        absoluteError > 5.0e-3 + 2.0e-3 * std::abs(expected))
      ++mismatches;
  }
  cudaFree(output.allocated);
  output = {};
  if (mismatches != 0) {
    std::fprintf(stderr,
                 "residual block correctness failed: mismatches=%zu "
                 "max_abs=%.9g max_rel=%.9g\n",
                 mismatches, maxAbsError, maxRelError);
    return 2;
  }
  if (std::getenv("LAUNCH_CHECK_ONLY")) {
    std::printf("launch_check,residual_conv_block,%s,correct,max_abs=%.9g,"
                "max_rel=%.9g\n",
                lowering, maxAbsError, maxRelError);
    return 0;
  }

  for (int i = 0; i < warmups; ++i) {
    _mlir_ciface_residual_conv_block(&output, &value, &weight1, &bias1,
                                     &weight2, &bias2);
    checkCuda(cudaDeviceSynchronize(), "warmup synchronize");
    cudaFree(output.allocated);
    output = {};
  }

  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop)");
  std::vector<double> deviceTimes;
  std::vector<double> wallTimes;
  for (int i = 0; i < runs; ++i) {
    checkCuda(cudaEventRecord(startEvent), "record start");
    auto wallStart = std::chrono::steady_clock::now();
    _mlir_ciface_residual_conv_block(&output, &value, &weight1, &bias1,
                                     &weight2, &bias2);
    checkCuda(cudaEventRecord(stopEvent), "record stop");
    checkCuda(cudaEventSynchronize(stopEvent), "event synchronize");
    auto wallStop = std::chrono::steady_clock::now();
    float elapsed = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsed, startEvent, stopEvent),
              "cudaEventElapsedTime");
    deviceTimes.push_back(elapsed);
    wallTimes.push_back(
        std::chrono::duration<double, std::milli>(wallStop - wallStart)
            .count());
    cudaFree(output.allocated);
    output = {};
  }

  std::printf("kind,name,backend,runs,p10_ms,p50_ms,p90_ms,wall_p50_ms,"
              "max_abs,max_rel\n");
  std::printf("result,residual_conv_block,%s,%d,%.9f,%.9f,%.9f,%.9f,%.9g,"
              "%.9g\n",
              lowering, runs, percentile(deviceTimes, 0.10),
              percentile(deviceTimes, 0.50), percentile(deviceTimes, 0.90),
              percentile(wallTimes, 0.50), maxAbsError, maxRelError);

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  cudaFree(bias2Data);
  cudaFree(weight2Data);
  cudaFree(bias1Data);
  cudaFree(weight1Data);
  cudaFree(valueData);
  return 0;
}
