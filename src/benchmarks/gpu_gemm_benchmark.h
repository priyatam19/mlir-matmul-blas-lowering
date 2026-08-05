#ifndef MLIR_TUTORIAL_GPU_GEMM_BENCHMARK_H
#define MLIR_TUTORIAL_GPU_GEMM_BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

template <typename T, int Rank> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[Rank];
  int64_t strides[Rank];
};

using CompiledGemm = void (*)(MemRefDescriptor<float, 2> *,
                              MemRefDescriptor<float, 2> *,
                              MemRefDescriptor<float, 2> *);

inline void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

inline void checkCublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "%s failed: cuBLAS status %d\n", operation,
                 static_cast<int>(status));
    std::exit(1);
  }
}

inline int envInt(const char *name, int defaultValue) {
  const char *value = std::getenv(name);
  if (!value)
    return defaultValue;
  int parsed = std::atoi(value);
  return parsed > 0 ? parsed : defaultValue;
}

struct TimingStats {
  double p10;
  double p50;
  double p90;
};

inline TimingStats summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  auto percentile = [&](double fraction) {
    size_t index = static_cast<size_t>((values.size() - 1) * fraction);
    return values[index];
  };
  return {percentile(0.10), percentile(0.50), percentile(0.90)};
}

template <int M, int K, int N>
int runGpuGemmBenchmark(const char *name, CompiledGemm compiledGemm) {
  const int warmups = envInt("WARMUPS", 10);
  const int runs = envInt("RUNS", 50);
  const int blockM = envInt("BLOCK_M", 8);
  const int blockN = envInt("BLOCK_N", 32);
  const int contractionBlockM = envInt("CONTRACTION_BLOCK_M", blockM);
  const int contractionBlockN = envInt("CONTRACTION_BLOCK_N", blockN);
  const int contractionBlockK = envInt("CONTRACTION_BLOCK_K", 16);
  const int contractionThreads = envInt("CONTRACTION_THREADS", 256);
  const int contractionStages = envInt("CONTRACTION_STAGES", 1);
  const char *lowering = std::getenv("GPU_LOWERING");
  if (!lowering)
    lowering = "block-thread";
  const char *mathMode = std::getenv("GPU_MATH_MODE");
  const bool tf32Mode = std::string(lowering) == "tensorcore-tf32" ||
                        (mathMode && std::string(mathMode) == "tf32");

  constexpr size_t aElements = static_cast<size_t>(M) * K;
  constexpr size_t bElements = static_cast<size_t>(K) * N;
  constexpr size_t cElements = static_cast<size_t>(M) * N;
  constexpr size_t aBytes = aElements * sizeof(float);
  constexpr size_t bBytes = bElements * sizeof(float);
  constexpr size_t cBytes = cElements * sizeof(float);

  float *aData = nullptr;
  float *bData = nullptr;
  float *cData = nullptr;
  float *referenceData = nullptr;
  float *pedanticReferenceData = nullptr;
  checkCuda(cudaMallocManaged(&aData, aBytes), "cudaMallocManaged(A)");
  checkCuda(cudaMallocManaged(&bData, bBytes), "cudaMallocManaged(B)");
  checkCuda(cudaMallocManaged(&cData, cBytes), "cudaMallocManaged(C)");
  checkCuda(cudaMallocManaged(&referenceData, cBytes),
            "cudaMallocManaged(reference)");
  checkCuda(cudaMallocManaged(&pedanticReferenceData, cBytes),
            "cudaMallocManaged(pedantic reference)");

  for (size_t i = 0; i < aElements; ++i)
    aData[i] = 0.001f * static_cast<float>(i % 100 + 1);
  for (size_t i = 0; i < bElements; ++i)
    bData[i] = 0.001f * static_cast<float>((i + 3) % 100 + 1);

  MemRefDescriptor<float, 2> a = {aData, aData, 0, {M, K}, {K, 1}};
  MemRefDescriptor<float, 2> b = {bData, bData, 0, {K, N}, {N, 1}};
  MemRefDescriptor<float, 2> c = {cData, cData, 0, {M, N}, {N, 1}};

  int device = 0;
  checkCuda(cudaGetDevice(&device), "cudaGetDevice");
  auto prefetchInputs = [&] {
    checkCuda(cudaMemPrefetchAsync(aData, aBytes, device), "prefetch A");
    checkCuda(cudaMemPrefetchAsync(bData, bBytes, device), "prefetch B");
    checkCuda(cudaMemPrefetchAsync(cData, cBytes, device), "prefetch C");
    checkCuda(cudaMemPrefetchAsync(referenceData, cBytes, device),
              "prefetch reference");
    checkCuda(cudaMemPrefetchAsync(pedanticReferenceData, cBytes, device),
              "prefetch pedantic reference");
    checkCuda(cudaDeviceSynchronize(), "prefetch synchronize");
  };
  prefetchInputs();

  cublasHandle_t cublas;
  checkCublas(cublasCreate(&cublas), "cublasCreate");
  checkCublas(cublasSetMathMode(
                  cublas, tf32Mode ? CUBLAS_TF32_TENSOR_OP_MATH
                                   : CUBLAS_PEDANTIC_MATH),
              "cublasSetMathMode");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto runCublas = [&](float *destination) {
    checkCublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            bData, N, aData, K, &beta, destination, N),
                "cublasSgemm");
  };

  checkCuda(cudaMemset(cData, 0, cBytes), "clear C for correctness");
  compiledGemm(&a, &b, &c);
  checkCuda(cudaDeviceSynchronize(), "compiled correctness synchronize");
  runCublas(referenceData);
  checkCuda(cudaDeviceSynchronize(), "cuBLAS correctness synchronize");

  double pedanticMaxAbsDrift = 0.0;
  double pedanticMaxRelDrift = 0.0;
  double pedanticSquaredDrift = 0.0;
  double pedanticSquaredReference = 0.0;
  if (tf32Mode) {
    checkCublas(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH),
                "cublasSetMathMode(pedantic reference)");
    runCublas(pedanticReferenceData);
    checkCuda(cudaDeviceSynchronize(),
              "cuBLAS pedantic correctness synchronize");
    checkCublas(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH),
                "cublasSetMathMode(restore TF32)");
    for (size_t i = 0; i < cElements; ++i) {
      double expected = pedanticReferenceData[i];
      double absoluteDrift = std::abs(static_cast<double>(cData[i]) - expected);
      double relativeDrift =
          absoluteDrift / std::max(std::abs(expected), 1.0e-6);
      pedanticMaxAbsDrift = std::max(pedanticMaxAbsDrift, absoluteDrift);
      pedanticMaxRelDrift = std::max(pedanticMaxRelDrift, relativeDrift);
      pedanticSquaredDrift += absoluteDrift * absoluteDrift;
      pedanticSquaredReference += expected * expected;
    }
  }
  double pedanticRelativeL2Drift = std::sqrt(
      pedanticSquaredDrift / std::max(pedanticSquaredReference, 1.0e-30));

  double maxAbsError = 0.0;
  double maxRelError = 0.0;
  double squaredError = 0.0;
  double squaredReference = 0.0;
  size_t mismatches = 0;
  const double absoluteTolerance = tf32Mode ? 1.0e-2 : 1.0e-4;
  const double relativeTolerance = tf32Mode ? 1.0e-2 : 1.0e-4;
  for (size_t i = 0; i < cElements; ++i) {
    double expected = referenceData[i];
    double actual = cData[i];
    double absoluteError = std::abs(actual - expected);
    double relativeError = absoluteError / std::max(std::abs(expected), 1.0e-6);
    maxAbsError = std::max(maxAbsError, absoluteError);
    maxRelError = std::max(maxRelError, relativeError);
    squaredError += absoluteError * absoluteError;
    squaredReference += expected * expected;
    if (!std::isfinite(actual) ||
        absoluteError >
            absoluteTolerance + relativeTolerance * std::abs(expected))
      ++mismatches;
  }
  double relativeL2 =
      std::sqrt(squaredError / std::max(squaredReference, 1.0e-30));
  const double relativeL2Tolerance = tf32Mode ? 5.0e-3 : 1.0e-5;
  if (relativeL2 > relativeL2Tolerance)
    ++mismatches;
  if (mismatches != 0) {
    std::fprintf(stderr,
                 "%s correctness failed: mismatches=%zu max_abs=%.9g "
                 "max_rel=%.9g rel_l2=%.9g\n",
                 name, mismatches, maxAbsError, maxRelError, relativeL2);
    return 2;
  }

  if (std::getenv("LAUNCH_CHECK_ONLY")) {
    std::printf("launch_check,%s,%s,correct,max_abs=%.9g,max_rel=%.9g,"
                "rel_l2=%.9g\n",
                name, lowering, maxAbsError, maxRelError, relativeL2);
    return 0;
  }

  // CPU validation above migrates managed pages, so return them to the GPU.
  prefetchInputs();

  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start)");
  checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop)");

  auto collect = [&](auto &&operation) {
    for (int i = 0; i < warmups; ++i) {
      checkCuda(cudaMemsetAsync(cData, 0, cBytes), "warmup clear C");
      operation();
    }
    checkCuda(cudaDeviceSynchronize(), "warmup synchronize");

    std::vector<double> deviceMilliseconds;
    std::vector<double> wallMilliseconds;
    deviceMilliseconds.reserve(runs);
    wallMilliseconds.reserve(runs);
    for (int i = 0; i < runs; ++i) {
      checkCuda(cudaMemsetAsync(cData, 0, cBytes), "timed clear C");
      checkCuda(cudaDeviceSynchronize(), "clear synchronize");
      checkCuda(cudaEventRecord(startEvent), "record start");
      auto wallStart = std::chrono::steady_clock::now();
      operation();
      checkCuda(cudaEventRecord(stopEvent), "record stop");
      checkCuda(cudaEventSynchronize(stopEvent), "event synchronize");
      auto wallStop = std::chrono::steady_clock::now();

      float elapsed = 0.0f;
      checkCuda(cudaEventElapsedTime(&elapsed, startEvent, stopEvent),
                "cudaEventElapsedTime");
      deviceMilliseconds.push_back(elapsed);
      wallMilliseconds.push_back(
          std::chrono::duration<double, std::milli>(wallStop - wallStart)
              .count());
    }
    return std::pair<TimingStats, TimingStats>{summarize(deviceMilliseconds),
                                               summarize(wallMilliseconds)};
  };

  auto compiledStats = collect([&] { compiledGemm(&a, &b, &c); });
  auto cublasStats = collect([&] { runCublas(cData); });
  double operations = 2.0 * static_cast<double>(M) * K * N;
  auto gflops = [&](double milliseconds) {
    return operations / (milliseconds * 1.0e6);
  };

  std::printf("kind,name,backend,m,k,n,block_m,block_n,block_k,threads,stages,"
              "runs,p10_ms,p50_ms,"
              "p90_ms,wall_p50_ms,gflops,max_abs,max_rel,rel_l2,"
              "pedantic_max_abs_drift,pedantic_max_rel_drift,"
              "pedantic_rel_l2_drift\n");
  std::printf("result,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,"
              "%.3f,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
              name, lowering, M, K, N, contractionBlockM, contractionBlockN,
              contractionBlockK, contractionThreads, contractionStages, runs,
              compiledStats.first.p10, compiledStats.first.p50,
              compiledStats.first.p90, compiledStats.second.p50,
              gflops(compiledStats.first.p50), maxAbsError, maxRelError,
              relativeL2, pedanticMaxAbsDrift, pedanticMaxRelDrift,
              pedanticRelativeL2Drift);
  std::printf("result,%s,%s,%d,%d,%d,0,0,0,0,0,%d,%.9f,%.9f,%.9f,%.9f,%.3f,0,0,"
              "0,0,0,0\n",
              name, tf32Mode ? "cublas-tf32" : "cublas-pedantic-fp32", M, K, N,
              runs, cublasStats.first.p10, cublasStats.first.p50,
              cublasStats.first.p90, cublasStats.second.p50,
              gflops(cublasStats.first.p50));

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  cublasDestroy(cublas);
  cudaFree(pedanticReferenceData);
  cudaFree(referenceData);
  cudaFree(cData);
  cudaFree(bData);
  cudaFree(aData);
  return 0;
}

#endif // MLIR_TUTORIAL_GPU_GEMM_BENCHMARK_H
