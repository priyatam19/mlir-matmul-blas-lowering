#include "TutorialGpuRuntime.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include <cublas_v2.h>

namespace {

void checkCublas(cublasStatus_t status, const char *operation) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return;
  std::fprintf(stderr, "%s failed with cuBLAS status %d\n", operation,
               static_cast<int>(status));
  std::abort();
}

cublasHandle_t getHandle() {
  static cublasHandle_t handle;
  static std::once_flag once;
  std::call_once(once, [] {
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH),
                "cublasSetMathMode");
  });
  return handle;
}

bool debugEnabled() {
  static bool enabled = std::getenv("TUTORIAL_GPU_RUNTIME_DEBUG") != nullptr;
  return enabled;
}

void require(bool condition, const char *message) {
  if (condition)
    return;
  std::fprintf(stderr, "tutorial GPU runtime: %s\n", message);
  std::abort();
}

} // namespace

#define TUTORIAL_MEMREF3_ARGS(name)                                            \
  float *name##Allocated, float *name##Aligned, int64_t name##Offset,          \
      int64_t name##Size0, int64_t name##Size1, int64_t name##Size2,           \
      int64_t name##Stride0, int64_t name##Stride1, int64_t name##Stride2

extern "C" void
tutorial_cublas_sgemm_strided_batched_f32(TUTORIAL_MEMREF3_ARGS(lhs),
                                          TUTORIAL_MEMREF3_ARGS(rhs),
                                          TUTORIAL_MEMREF3_ARGS(output)) {
  (void)lhsAllocated;
  (void)rhsAllocated;
  (void)outputAllocated;
  require(lhsSize0 == rhsSize0 && lhsSize0 == outputSize0,
          "batch dimensions do not match");
  require(lhsSize1 == outputSize1 && rhsSize2 == outputSize2 &&
              lhsSize2 == rhsSize1,
          "batch matmul dimensions do not match");
  require(lhsStride2 == 1 && rhsStride2 == 1 && outputStride2 == 1,
          "cuBLAS batch matmul requires unit innermost strides");

  float *lhs = lhsAligned + lhsOffset;
  float *rhs = rhsAligned + rhsOffset;
  float *output = outputAligned + outputOffset;
  const float alpha = 1.0f;
  const float beta = 1.0f;

  checkCublas(cublasSgemmStridedBatched(
                  getHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
                  static_cast<int>(outputSize2), static_cast<int>(outputSize1),
                  static_cast<int>(lhsSize2), &alpha, rhs,
                  static_cast<int>(rhsStride1), rhsStride0, lhs,
                  static_cast<int>(lhsStride1), lhsStride0, &beta, output,
                  static_cast<int>(outputStride1), outputStride0,
                  static_cast<int>(outputSize0)),
              "cublasSgemmStridedBatched");

  if (debugEnabled()) {
    static std::atomic<uint64_t> calls{0};
    uint64_t call = ++calls;
    std::fprintf(stderr,
                 "tutorial_gpu_runtime,op=cublas_batch_matmul,call=%llu,"
                 "batch=%lld,m=%lld,k=%lld,n=%lld\n",
                 static_cast<unsigned long long>(call),
                 static_cast<long long>(outputSize0),
                 static_cast<long long>(outputSize1),
                 static_cast<long long>(lhsSize2),
                 static_cast<long long>(outputSize2));
  }
}

#undef TUTORIAL_MEMREF3_ARGS
