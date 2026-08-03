#ifndef MLIR_TUTORIAL_GPU_CONV_BENCHMARK_H
#define MLIR_TUTORIAL_GPU_CONV_BENCHMARK_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

template <typename T, int Rank> struct ConvMemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

using CompiledConv = void (*)(ConvMemRefDescriptor<float, 4> *,
                              ConvMemRefDescriptor<float, 4> *,
                              ConvMemRefDescriptor<float, 4> *);

inline void checkConvCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

inline void checkConvCudnn(cudnnStatus_t status, const char *operation) {
  if (status != CUDNN_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudnnGetErrorString(status));
    std::exit(1);
  }
}

inline int convEnvInt(const char *name, int defaultValue) {
  const char *value = std::getenv(name);
  if (!value)
    return defaultValue;
  int parsed = std::atoi(value);
  return parsed > 0 ? parsed : defaultValue;
}

struct ConvTimingStats {
  double p10;
  double p50;
  double p90;
};

inline ConvTimingStats summarizeConv(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  auto percentile = [&](double fraction) {
    return values[static_cast<size_t>((values.size() - 1) * fraction)];
  };
  return {percentile(0.10), percentile(0.50), percentile(0.90)};
}

template <int N, int C, int H, int W, int F, int KH, int KW, int OH, int OW,
          int StrideH = 1, int StrideW = 1, int DilationH = 1,
          int DilationW = 1>
int runGpuConvBenchmark(const char *name, CompiledConv compiledConv) {
  const int warmups = convEnvInt("WARMUPS", 10);
  const int runs = convEnvInt("RUNS", 50);
  const int threads = convEnvInt("CONV_THREADS", 256);
  const char *lowering = std::getenv("GPU_LOWERING");
  if (!lowering)
    lowering = "block-thread";

  constexpr size_t inputElements = static_cast<size_t>(N) * C * H * W;
  constexpr size_t filterElements = static_cast<size_t>(F) * C * KH * KW;
  constexpr size_t outputElements = static_cast<size_t>(N) * F * OH * OW;
  constexpr size_t inputBytes = inputElements * sizeof(float);
  constexpr size_t filterBytes = filterElements * sizeof(float);
  constexpr size_t outputBytes = outputElements * sizeof(float);

  float *inputData = nullptr;
  float *filterData = nullptr;
  float *outputData = nullptr;
  float *referenceData = nullptr;
  checkConvCuda(cudaMallocManaged(&inputData, inputBytes),
                "cudaMallocManaged(input)");
  checkConvCuda(cudaMallocManaged(&filterData, filterBytes),
                "cudaMallocManaged(filter)");
  checkConvCuda(cudaMallocManaged(&outputData, outputBytes),
                "cudaMallocManaged(output)");
  checkConvCuda(cudaMallocManaged(&referenceData, outputBytes),
                "cudaMallocManaged(reference)");

  for (size_t i = 0; i < inputElements; ++i)
    inputData[i] = 0.002f * static_cast<float>(static_cast<int>(i % 101) - 50);
  for (size_t i = 0; i < filterElements; ++i)
    filterData[i] =
        0.001f * static_cast<float>(static_cast<int>((i + 17) % 97) - 48);

  ConvMemRefDescriptor<float, 4> input = {
      inputData, inputData, 0, {N, C, H, W}, {C * H * W, H * W, W, 1}};
  ConvMemRefDescriptor<float, 4> filter = {
      filterData, filterData, 0, {F, C, KH, KW}, {C * KH * KW, KH * KW, KW, 1}};
  ConvMemRefDescriptor<float, 4> output = {
      outputData, outputData, 0, {N, F, OH, OW}, {F * OH * OW, OH * OW, OW, 1}};

  int device = 0;
  checkConvCuda(cudaGetDevice(&device), "cudaGetDevice");
  auto prefetch = [&] {
    checkConvCuda(cudaMemPrefetchAsync(inputData, inputBytes, device),
                  "prefetch input");
    checkConvCuda(cudaMemPrefetchAsync(filterData, filterBytes, device),
                  "prefetch filter");
    checkConvCuda(cudaMemPrefetchAsync(outputData, outputBytes, device),
                  "prefetch output");
    checkConvCuda(cudaMemPrefetchAsync(referenceData, outputBytes, device),
                  "prefetch reference");
    checkConvCuda(cudaDeviceSynchronize(), "prefetch synchronize");
  };
  prefetch();

  cudnnHandle_t cudnn;
  cudnnTensorDescriptor_t inputDescriptor;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnTensorDescriptor_t outputDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  checkConvCudnn(cudnnCreate(&cudnn), "cudnnCreate");
  checkConvCudnn(cudnnCreateTensorDescriptor(&inputDescriptor),
                 "cudnnCreateTensorDescriptor(input)");
  checkConvCudnn(cudnnCreateFilterDescriptor(&filterDescriptor),
                 "cudnnCreateFilterDescriptor");
  checkConvCudnn(cudnnCreateTensorDescriptor(&outputDescriptor),
                 "cudnnCreateTensorDescriptor(output)");
  checkConvCudnn(cudnnCreateConvolutionDescriptor(&convolutionDescriptor),
                 "cudnnCreateConvolutionDescriptor");
  checkConvCudnn(cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, N, C, H, W),
                 "cudnnSetTensor4dDescriptor(input)");
  checkConvCudnn(cudnnSetFilter4dDescriptor(filterDescriptor, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, F, C, KH, KW),
                 "cudnnSetFilter4dDescriptor");
  checkConvCudnn(cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, N, F, OH, OW),
                 "cudnnSetTensor4dDescriptor(output)");
  checkConvCudnn(cudnnSetConvolution2dDescriptor(
                     convolutionDescriptor, 0, 0, StrideH, StrideW, DilationH,
                     DilationW, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                 "cudnnSetConvolution2dDescriptor");
  checkConvCudnn(
      cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_FMA_MATH),
      "cudnnSetConvolutionMathType");

  std::array<cudnnConvolutionFwdAlgoPerf_t, CUDNN_CONVOLUTION_FWD_ALGO_COUNT>
      algorithms{};
  int algorithmCount = 0;
  checkConvCudnn(cudnnGetConvolutionForwardAlgorithm_v7(
                     cudnn, inputDescriptor, filterDescriptor,
                     convolutionDescriptor, outputDescriptor,
                     static_cast<int>(algorithms.size()), &algorithmCount,
                     algorithms.data()),
                 "cudnnGetConvolutionForwardAlgorithm_v7");
  cudnnConvolutionFwdAlgo_t algorithm =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t workspaceBytes = 0;
  bool foundAlgorithm = false;
  constexpr size_t workspaceLimit = 256ULL * 1024ULL * 1024ULL;
  for (int i = 0; i < algorithmCount; ++i) {
    if (algorithms[i].status == CUDNN_STATUS_SUCCESS &&
        algorithms[i].memory <= workspaceLimit) {
      algorithm = algorithms[i].algo;
      workspaceBytes = algorithms[i].memory;
      foundAlgorithm = true;
      break;
    }
  }
  if (!foundAlgorithm) {
    std::fprintf(stderr, "%s: no cuDNN algorithm fits the workspace limit\n",
                 name);
    return 3;
  }
  void *workspace = nullptr;
  if (workspaceBytes)
    checkConvCuda(cudaMalloc(&workspace, workspaceBytes),
                  "cudaMalloc(cuDNN workspace)");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto runCudnn = [&](float *destination) {
    checkConvCudnn(cudnnConvolutionForward(
                       cudnn, &alpha, inputDescriptor, inputData,
                       filterDescriptor, filterData, convolutionDescriptor,
                       algorithm, workspace, workspaceBytes, &beta,
                       outputDescriptor, destination),
                   "cudnnConvolutionForward");
  };

  checkConvCuda(cudaMemset(outputData, 0, outputBytes),
                "clear output for correctness");
  compiledConv(&input, &filter, &output);
  checkConvCuda(cudaDeviceSynchronize(), "compiled correctness synchronize");
  runCudnn(referenceData);
  checkConvCuda(cudaDeviceSynchronize(), "cuDNN correctness synchronize");

  double maxAbsError = 0.0;
  double maxRelError = 0.0;
  size_t mismatches = 0;
  constexpr double absoluteTolerance = 5.0e-3;
  constexpr double relativeTolerance = 2.0e-3;
  for (size_t i = 0; i < outputElements; ++i) {
    double expected = referenceData[i];
    double actual = outputData[i];
    double absoluteError = std::abs(actual - expected);
    double relativeError = absoluteError / std::max(std::abs(expected), 1.0e-6);
    maxAbsError = std::max(maxAbsError, absoluteError);
    maxRelError = std::max(maxRelError, relativeError);
    if (!std::isfinite(actual) ||
        absoluteError >
            absoluteTolerance + relativeTolerance * std::abs(expected))
      ++mismatches;
  }
  if (mismatches != 0) {
    std::fprintf(stderr,
                 "%s correctness failed: mismatches=%zu max_abs=%.9g "
                 "max_rel=%.9g\n",
                 name, mismatches, maxAbsError, maxRelError);
    return 2;
  }

  if (std::getenv("LAUNCH_CHECK_ONLY")) {
    std::printf("launch_check,%s,%s,correct,max_abs=%.9g,max_rel=%.9g\n", name,
                lowering, maxAbsError, maxRelError);
    return 0;
  }

  prefetch();
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  checkConvCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start)");
  checkConvCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop)");

  auto collect = [&](auto &&operation) {
    for (int i = 0; i < warmups; ++i) {
      checkConvCuda(cudaMemsetAsync(outputData, 0, outputBytes),
                    "warmup clear output");
      operation();
    }
    checkConvCuda(cudaDeviceSynchronize(), "warmup synchronize");

    std::vector<double> deviceMilliseconds;
    std::vector<double> wallMilliseconds;
    deviceMilliseconds.reserve(runs);
    wallMilliseconds.reserve(runs);
    for (int i = 0; i < runs; ++i) {
      checkConvCuda(cudaMemsetAsync(outputData, 0, outputBytes),
                    "timed clear output");
      checkConvCuda(cudaDeviceSynchronize(), "clear synchronize");
      checkConvCuda(cudaEventRecord(startEvent), "record start");
      auto wallStart = std::chrono::steady_clock::now();
      operation();
      checkConvCuda(cudaEventRecord(stopEvent), "record stop");
      checkConvCuda(cudaEventSynchronize(stopEvent), "event synchronize");
      auto wallStop = std::chrono::steady_clock::now();
      float elapsed = 0.0f;
      checkConvCuda(cudaEventElapsedTime(&elapsed, startEvent, stopEvent),
                    "cudaEventElapsedTime");
      deviceMilliseconds.push_back(elapsed);
      wallMilliseconds.push_back(
          std::chrono::duration<double, std::milli>(wallStop - wallStart)
              .count());
    }
    return std::pair<ConvTimingStats, ConvTimingStats>{
        summarizeConv(deviceMilliseconds), summarizeConv(wallMilliseconds)};
  };

  auto compiledStats = collect([&] { compiledConv(&input, &filter, &output); });
  auto cudnnStats = collect([&] { runCudnn(outputData); });
  double operations = 2.0 * N * static_cast<double>(F) * OH * OW * C * KH * KW;
  auto gflops = [&](double milliseconds) {
    return operations / (milliseconds * 1.0e6);
  };

  std::printf("kind,name,backend,n,c,h,w,f,kh,kw,oh,ow,stride_h,stride_w,"
              "dilation_h,dilation_w,threads,runs,p10_ms,p50_ms,p90_ms,"
              "wall_p50_ms,gflops,max_abs,max_rel\n");
  std::printf("result,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
              "%.9f,%.9f,%.9f,%.9f,%.3f,%.9g,%.9g\n",
              name, lowering, N, C, H, W, F, KH, KW, OH, OW, StrideH, StrideW,
              DilationH, DilationW, threads, runs, compiledStats.first.p10,
              compiledStats.first.p50, compiledStats.first.p90,
              compiledStats.second.p50, gflops(compiledStats.first.p50),
              maxAbsError, maxRelError);
  std::printf("result,%s,cudnn-fp32,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
              "0,%d,%.9f,%.9f,%.9f,%.9f,%.3f,0,0\n",
              name, N, C, H, W, F, KH, KW, OH, OW, StrideH, StrideW, DilationH,
              DilationW, runs, cudnnStats.first.p10, cudnnStats.first.p50,
              cudnnStats.first.p90, cudnnStats.second.p50,
              gflops(cudnnStats.first.p50));

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  if (workspace)
    cudaFree(workspace);
  cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
  cudnnDestroyTensorDescriptor(outputDescriptor);
  cudnnDestroyFilterDescriptor(filterDescriptor);
  cudnnDestroyTensorDescriptor(inputDescriptor);
  cudnnDestroy(cudnn);
  cudaFree(referenceData);
  cudaFree(outputData);
  cudaFree(filterData);
  cudaFree(inputData);
  return 0;
}

#endif // MLIR_TUTORIAL_GPU_CONV_BENCHMARK_H
