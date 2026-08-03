#ifndef MLIR_TUTORIAL_GPU_BMM_BENCHMARK_H
#define MLIR_TUTORIAL_GPU_BMM_BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename T, int Rank> struct BmmMemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

using CompiledBmm = void (*)(BmmMemRefDescriptor<float, 3> *,
                             BmmMemRefDescriptor<float, 3> *,
                             BmmMemRefDescriptor<float, 3> *);

inline void checkBmmCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

inline void checkBmmCublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: cuBLAS status %d\n", operation,
                 static_cast<int>(status));
    std::exit(1);
  }
}

inline int bmmEnvInt(const char *name, int defaultValue) {
  const char *value = std::getenv(name);
  if (!value)
    return defaultValue;
  int parsed = std::atoi(value);
  return parsed > 0 ? parsed : defaultValue;
}

struct BmmTimingStats {
  double p10;
  double p50;
  double p90;
};

inline BmmTimingStats summarizeBmm(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  auto percentile = [&](double fraction) {
    return values[static_cast<size_t>((values.size() - 1) * fraction)];
  };
  return {percentile(0.10), percentile(0.50), percentile(0.90)};
}

template <int Batch, int M, int K, int N>
int runGpuBmmBenchmark(const char *name, CompiledBmm compiledBmm) {
  const int warmups = bmmEnvInt("WARMUPS", 10);
  const int runs = bmmEnvInt("RUNS", 50);
  const int blockM = bmmEnvInt("BLOCK_M", 8);
  const int blockN = bmmEnvInt("BLOCK_N", 32);
  const char *lowering = std::getenv("GPU_LOWERING");
  if (!lowering)
    lowering = "block-thread";

  constexpr size_t aElements = static_cast<size_t>(Batch) * M * K;
  constexpr size_t bElements = static_cast<size_t>(Batch) * K * N;
  constexpr size_t cElements = static_cast<size_t>(Batch) * M * N;
  constexpr size_t aBytes = aElements * sizeof(float);
  constexpr size_t bBytes = bElements * sizeof(float);
  constexpr size_t cBytes = cElements * sizeof(float);

  float *aData = nullptr;
  float *bData = nullptr;
  float *cData = nullptr;
  float *referenceData = nullptr;
  checkBmmCuda(cudaMallocManaged(&aData, aBytes), "cudaMallocManaged(A)");
  checkBmmCuda(cudaMallocManaged(&bData, bBytes), "cudaMallocManaged(B)");
  checkBmmCuda(cudaMallocManaged(&cData, cBytes), "cudaMallocManaged(C)");
  checkBmmCuda(cudaMallocManaged(&referenceData, cBytes),
               "cudaMallocManaged(reference)");

  for (size_t i = 0; i < aElements; ++i)
    aData[i] = 0.001f * static_cast<float>(i % 101 + 1);
  for (size_t i = 0; i < bElements; ++i)
    bData[i] = 0.001f * static_cast<float>((i + 7) % 103 + 1);

  BmmMemRefDescriptor<float, 3> a = {
      aData, aData, 0, {Batch, M, K}, {M * K, K, 1}};
  BmmMemRefDescriptor<float, 3> b = {
      bData, bData, 0, {Batch, K, N}, {K * N, N, 1}};
  BmmMemRefDescriptor<float, 3> c = {
      cData, cData, 0, {Batch, M, N}, {M * N, N, 1}};

  int device = 0;
  checkBmmCuda(cudaGetDevice(&device), "cudaGetDevice");
  auto prefetch = [&] {
    checkBmmCuda(cudaMemPrefetchAsync(aData, aBytes, device), "prefetch A");
    checkBmmCuda(cudaMemPrefetchAsync(bData, bBytes, device), "prefetch B");
    checkBmmCuda(cudaMemPrefetchAsync(cData, cBytes, device), "prefetch C");
    checkBmmCuda(cudaMemPrefetchAsync(referenceData, cBytes, device),
                 "prefetch reference");
    checkBmmCuda(cudaDeviceSynchronize(), "prefetch synchronize");
  };
  prefetch();

  cublasHandle_t cublas;
  checkBmmCublas(cublasCreate(&cublas), "cublasCreate");
  checkBmmCublas(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH),
                 "cublasSetMathMode");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto runCublas = [&](float *destination) {
    checkBmmCublas(cublasSgemmStridedBatched(
                       cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, bData,
                       N, static_cast<long long>(K) * N, aData, K,
                       static_cast<long long>(M) * K, &beta, destination, N,
                       static_cast<long long>(M) * N, Batch),
                   "cublasSgemmStridedBatched");
  };

  checkBmmCuda(cudaMemset(cData, 0, cBytes), "clear C for correctness");
  compiledBmm(&a, &b, &c);
  checkBmmCuda(cudaDeviceSynchronize(), "compiled correctness synchronize");
  runCublas(referenceData);
  checkBmmCuda(cudaDeviceSynchronize(), "cuBLAS correctness synchronize");

  double maxAbsError = 0.0;
  double maxRelError = 0.0;
  size_t mismatches = 0;
  constexpr double absoluteTolerance = 2.0e-3;
  constexpr double relativeTolerance = 2.0e-3;
  for (size_t i = 0; i < cElements; ++i) {
    double expected = referenceData[i];
    double actual = cData[i];
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
  checkBmmCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start)");
  checkBmmCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop)");

  auto collect = [&](auto &&operation) {
    for (int i = 0; i < warmups; ++i) {
      checkBmmCuda(cudaMemsetAsync(cData, 0, cBytes), "warmup clear C");
      operation();
    }
    checkBmmCuda(cudaDeviceSynchronize(), "warmup synchronize");

    std::vector<double> deviceMilliseconds;
    std::vector<double> wallMilliseconds;
    deviceMilliseconds.reserve(runs);
    wallMilliseconds.reserve(runs);
    for (int i = 0; i < runs; ++i) {
      checkBmmCuda(cudaMemsetAsync(cData, 0, cBytes), "timed clear C");
      checkBmmCuda(cudaDeviceSynchronize(), "clear synchronize");
      checkBmmCuda(cudaEventRecord(startEvent), "record start");
      auto wallStart = std::chrono::steady_clock::now();
      operation();
      checkBmmCuda(cudaEventRecord(stopEvent), "record stop");
      checkBmmCuda(cudaEventSynchronize(stopEvent), "event synchronize");
      auto wallStop = std::chrono::steady_clock::now();
      float elapsed = 0.0f;
      checkBmmCuda(cudaEventElapsedTime(&elapsed, startEvent, stopEvent),
                   "cudaEventElapsedTime");
      deviceMilliseconds.push_back(elapsed);
      wallMilliseconds.push_back(
          std::chrono::duration<double, std::milli>(wallStop - wallStart)
              .count());
    }
    return std::pair<BmmTimingStats, BmmTimingStats>{
        summarizeBmm(deviceMilliseconds), summarizeBmm(wallMilliseconds)};
  };

  auto compiledStats = collect([&] { compiledBmm(&a, &b, &c); });
  auto cublasStats = collect([&] { runCublas(cData); });
  double operations = 2.0 * Batch * static_cast<double>(M) * K * N;
  auto gflops = [&](double milliseconds) {
    return operations / (milliseconds * 1.0e6);
  };

  std::printf("kind,name,backend,batch,m,k,n,block_m,block_n,runs,p10_ms,"
              "p50_ms,p90_ms,wall_p50_ms,gflops,max_abs,max_rel\n");
  std::printf("result,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,"
              "%.3f,%.9g,%.9g\n",
              name, lowering, Batch, M, K, N, blockM, blockN, runs,
              compiledStats.first.p10, compiledStats.first.p50,
              compiledStats.first.p90, compiledStats.second.p50,
              gflops(compiledStats.first.p50), maxAbsError, maxRelError);
  std::printf("result,%s,cublas-pedantic-fp32,%d,%d,%d,%d,0,0,%d,%.9f,"
              "%.9f,%.9f,%.9f,%.3f,0,0\n",
              name, Batch, M, K, N, runs, cublasStats.first.p10,
              cublasStats.first.p50, cublasStats.first.p90,
              cublasStats.second.p50, gflops(cublasStats.first.p50));

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  cublasDestroy(cublas);
  cudaFree(referenceData);
  cudaFree(cData);
  cudaFree(bData);
  cudaFree(aData);
  return 0;
}

#endif // MLIR_TUTORIAL_GPU_BMM_BENCHMARK_H
