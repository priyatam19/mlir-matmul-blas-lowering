// cuBLAS baseline: large_gemm (512x256) x (256x512) -> (512x512)
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

static void checkCuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

static void checkCublas(cublasStatus_t status, const char *what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << what << ": cuBLAS status " << static_cast<int>(status)
              << "\n";
    std::exit(1);
  }
}

int main() {
  constexpr int M = 512;
  constexpr int K = 256;
  constexpr int N = 512;
  constexpr int warmups = 10;
  constexpr int runs = 100;

  std::vector<float> hA(M * K);
  std::vector<float> hB(K * N);
  for (int i = 0; i < M * K; ++i)
    hA[i] = 0.001f * static_cast<float>(i % 100 + 1);
  for (int i = 0; i < K * N; ++i)
    hB[i] = 0.001f * static_cast<float>((i + 3) % 100 + 1);

  float *dA = nullptr;
  float *dB = nullptr;
  float *dC = nullptr;
  checkCuda(cudaMalloc(&dA, M * K * sizeof(float)), "cudaMalloc A");
  checkCuda(cudaMalloc(&dB, K * N * sizeof(float)), "cudaMalloc B");
  checkCuda(cudaMalloc(&dC, M * N * sizeof(float)), "cudaMalloc C");
  checkCuda(cudaMemcpy(dA, hA.data(), M * K * sizeof(float),
                       cudaMemcpyHostToDevice),
            "copy A");
  checkCuda(cudaMemcpy(dB, hB.data(), K * N * sizeof(float),
                       cudaMemcpyHostToDevice),
            "copy B");

  cublasHandle_t handle;
  checkCublas(cublasCreate(&handle), "cublasCreate");
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Row-major C = A(MxK) * B(KxN) is equivalent to column-major
  // C^T = B^T(NxK) * A^T(KxM).
  auto runGemm = [&] {
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            dB, N, dA, K, &beta, dC, N),
                "cublasSgemm");
  };

  for (int i = 0; i < warmups; ++i)
    runGemm();
  checkCuda(cudaDeviceSynchronize(), "warmup sync");

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; ++i)
    runGemm();
  checkCuda(cudaDeviceSynchronize(), "timed sync");
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::vector<float> hC(M * N);
  checkCuda(cudaMemcpy(hC.data(), dC, M * N * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "copy C");

  std::cout << "large_gemm cuBLAS GPU avg inference time: " << std::fixed
            << std::setprecision(9) << (elapsed.count() / runs) << " sec\n";
  std::cout << "checksum sample: " << std::fixed << std::setprecision(6)
            << hC[0] << " " << hC[M * (N / 2) + (N / 2)] << " "
            << hC[M * N - 1] << "\n";

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
