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
void __real_free(void *ptr);

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

template <typename T, int Rank> struct AttentionMemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

extern "C" void
_mlir_ciface_attention_block(AttentionMemRefDescriptor<float, 3> *output,
                             AttentionMemRefDescriptor<float, 3> *query,
                             AttentionMemRefDescriptor<float, 3> *key,
                             AttentionMemRefDescriptor<float, 3> *value);

namespace {

constexpr int Batch = 12;
constexpr int Sequence = 128;
constexpr int HeadDim = 64;

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

size_t inputIndex(int batch, int sequence, int dim) {
  return (static_cast<size_t>(batch) * Sequence + sequence) * HeadDim + dim;
}

std::vector<float> referenceAttention(const float *query, const float *key,
                                      const float *value) {
  std::vector<float> output(static_cast<size_t>(Batch) * Sequence * HeadDim);
  std::vector<float> probabilities(Sequence);
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < Sequence; ++row) {
      float maximum = -INFINITY;
      for (int column = 0; column < Sequence; ++column) {
        float score = 0.0f;
        for (int dim = 0; dim < HeadDim; ++dim) {
          score += query[inputIndex(batch, row, dim)] *
                   key[inputIndex(batch, column, dim)];
        }
        probabilities[column] = score * 0.125f;
        maximum = std::max(maximum, probabilities[column]);
      }
      float denominator = 0.0f;
      for (float &probability : probabilities) {
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      for (float &probability : probabilities)
        probability /= denominator;
      for (int dim = 0; dim < HeadDim; ++dim) {
        float result = 0.0f;
        for (int column = 0; column < Sequence; ++column) {
          result +=
              probabilities[column] * value[inputIndex(batch, column, dim)];
        }
        output[inputIndex(batch, row, dim)] = result;
      }
    }
  }
  return output;
}

std::vector<float> loadPyTorchReference(const char *path, size_t elements) {
  std::vector<float> output(elements);
  std::FILE *file = std::fopen(path, "rb");
  if (!file) {
    std::perror(path);
    std::exit(1);
  }
  size_t readElements =
      std::fread(output.data(), sizeof(float), output.size(), file);
  int trailingByte = std::fgetc(file);
  std::fclose(file);
  if (readElements != output.size() || trailingByte != EOF) {
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
  constexpr size_t elements = static_cast<size_t>(Batch) * Sequence * HeadDim;
  constexpr size_t bytes = elements * sizeof(float);

  float *queryData = nullptr;
  float *keyData = nullptr;
  float *valueData = nullptr;
  checkCuda(cudaMallocManaged(&queryData, bytes), "cudaMallocManaged(query)");
  checkCuda(cudaMallocManaged(&keyData, bytes), "cudaMallocManaged(key)");
  checkCuda(cudaMallocManaged(&valueData, bytes), "cudaMallocManaged(value)");
  for (size_t i = 0; i < elements; ++i) {
    queryData[i] = 0.001f * static_cast<float>(i % 101 + 1);
    keyData[i] = 0.001f * static_cast<float>((i + 11) % 103 + 1);
    valueData[i] = 0.001f * static_cast<float>((i + 17) % 107 + 1);
  }

  AttentionMemRefDescriptor<float, 3> query = {
      queryData,
      queryData,
      0,
      {Batch, Sequence, HeadDim},
      {Sequence * HeadDim, HeadDim, 1}};
  AttentionMemRefDescriptor<float, 3> key = {keyData,
                                             keyData,
                                             0,
                                             {Batch, Sequence, HeadDim},
                                             {Sequence * HeadDim, HeadDim, 1}};
  AttentionMemRefDescriptor<float, 3> value = {
      valueData,
      valueData,
      0,
      {Batch, Sequence, HeadDim},
      {Sequence * HeadDim, HeadDim, 1}};
  AttentionMemRefDescriptor<float, 3> output = {};

  int device = 0;
  checkCuda(cudaGetDevice(&device), "cudaGetDevice");
  checkCuda(cudaMemPrefetchAsync(queryData, bytes, device), "prefetch query");
  checkCuda(cudaMemPrefetchAsync(keyData, bytes, device), "prefetch key");
  checkCuda(cudaMemPrefetchAsync(valueData, bytes, device), "prefetch value");
  checkCuda(cudaDeviceSynchronize(), "prefetch synchronize");

  _mlir_ciface_attention_block(&output, &query, &key, &value);
  checkCuda(cudaDeviceSynchronize(), "correctness synchronize");
  std::vector<float> reference =
      referenceAttention(queryData, keyData, valueData);
  if (const char *referencePath = std::getenv("PYTORCH_REFERENCE"))
    reference = loadPyTorchReference(referencePath, elements);
  double maxAbsError = 0.0;
  double maxRelError = 0.0;
  size_t mismatches = 0;
  for (size_t i = 0; i < elements; ++i) {
    double actual = output.aligned[output.offset + i];
    double expected = reference[i];
    double absoluteError = std::abs(actual - expected);
    double relativeError = absoluteError / std::max(std::abs(expected), 1.0e-6);
    maxAbsError = std::max(maxAbsError, absoluteError);
    maxRelError = std::max(maxRelError, relativeError);
    if (!std::isfinite(actual) ||
        absoluteError > 2.0e-3 + 2.0e-3 * std::abs(expected))
      ++mismatches;
  }
  cudaFree(output.allocated);
  output = {};
  if (mismatches != 0) {
    std::fprintf(stderr,
                 "attention correctness failed: mismatches=%zu max_abs=%.9g "
                 "max_rel=%.9g\n",
                 mismatches, maxAbsError, maxRelError);
    return 2;
  }
  if (std::getenv("LAUNCH_CHECK_ONLY")) {
    std::printf("launch_check,attention_block,%s,correct,max_abs=%.9g,"
                "max_rel=%.9g\n",
                lowering, maxAbsError, maxRelError);
    return 0;
  }

  for (int i = 0; i < warmups; ++i) {
    _mlir_ciface_attention_block(&output, &query, &key, &value);
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
    _mlir_ciface_attention_block(&output, &query, &key, &value);
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
  std::printf("result,attention_block,%s,%d,%.9f,%.9f,%.9f,%.9f,%.9g,%.9g\n",
              lowering, runs, percentile(deviceTimes, 0.10),
              percentile(deviceTimes, 0.50), percentile(deviceTimes, 0.90),
              percentile(wallTimes, 0.50), maxAbsError, maxRelError);

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  cudaFree(valueData);
  cudaFree(keyData);
  cudaFree(queryData);
  return 0;
}
