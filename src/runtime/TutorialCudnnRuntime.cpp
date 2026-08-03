#include "TutorialGpuRuntime.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unordered_map>

#include <cuda_runtime_api.h>
#include <cudnn.h>

namespace {

void checkCuda(cudaError_t status, const char *operation) {
  if (status == cudaSuccess)
    return;
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudaGetErrorString(status));
  std::abort();
}

void checkCudnn(cudnnStatus_t status, const char *operation) {
  if (status == CUDNN_STATUS_SUCCESS)
    return;
  std::fprintf(stderr, "%s failed: %s\n", operation,
               cudnnGetErrorString(status));
  std::abort();
}

void require(bool condition, const char *message) {
  if (condition)
    return;
  std::fprintf(stderr, "tutorial cuDNN runtime: %s\n", message);
  std::abort();
}

int checkedInt(int64_t value, const char *description) {
  require(value > 0 && value <= INT_MAX, description);
  return static_cast<int>(value);
}

cudnnHandle_t getHandle() {
  static thread_local cudnnHandle_t handle = [] {
    cudnnHandle_t result;
    checkCudnn(cudnnCreate(&result), "cudnnCreate");
    return result;
  }();
  return handle;
}

bool debugEnabled() {
  static bool enabled = std::getenv("TUTORIAL_GPU_RUNTIME_DEBUG") != nullptr;
  return enabled;
}

size_t workspaceLimit() {
  constexpr size_t defaultLimit = 256ULL * 1024ULL * 1024ULL;
  const char *value = std::getenv("CUDNN_WORKSPACE_LIMIT_MB");
  if (!value)
    return defaultLimit;
  errno = 0;
  char *end = nullptr;
  unsigned long long megabytes = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' ||
      megabytes > SIZE_MAX / (1024ULL * 1024ULL)) {
    std::fprintf(stderr, "Invalid CUDNN_WORKSPACE_LIMIT_MB=%s\n", value);
    std::abort();
  }
  return static_cast<size_t>(megabytes * 1024ULL * 1024ULL);
}

using ConvKey = std::array<int64_t, 24>;

struct ConvKeyHash {
  size_t operator()(const ConvKey &key) const {
    size_t hash = 0;
    for (int64_t value : key)
      hash ^=
          std::hash<int64_t>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

struct ConvPlan {
  cudnnTensorDescriptor_t input = nullptr;
  cudnnFilterDescriptor_t filter = nullptr;
  cudnnTensorDescriptor_t output = nullptr;
  cudnnConvolutionDescriptor_t convolution = nullptr;
  cudnnConvolutionFwdAlgo_t algorithm =
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  void *workspace = nullptr;
  size_t workspaceBytes = 0;

  ~ConvPlan() {
    if (workspace)
      cudaFree(workspace);
    if (convolution)
      cudnnDestroyConvolutionDescriptor(convolution);
    if (output)
      cudnnDestroyTensorDescriptor(output);
    if (filter)
      cudnnDestroyFilterDescriptor(filter);
    if (input)
      cudnnDestroyTensorDescriptor(input);
  }
};

std::unique_ptr<ConvPlan> createPlan(const std::array<int, 4> &inputDims,
                                     const std::array<int, 4> &inputStrides,
                                     const std::array<int, 4> &filterDims,
                                     const std::array<int, 4> &outputDims,
                                     const std::array<int, 4> &outputStrides,
                                     int strideH, int strideW, int dilationH,
                                     int dilationW) {
  auto plan = std::make_unique<ConvPlan>();
  checkCudnn(cudnnCreateTensorDescriptor(&plan->input),
             "cudnnCreateTensorDescriptor(input)");
  checkCudnn(cudnnCreateFilterDescriptor(&plan->filter),
             "cudnnCreateFilterDescriptor");
  checkCudnn(cudnnCreateTensorDescriptor(&plan->output),
             "cudnnCreateTensorDescriptor(output)");
  checkCudnn(cudnnCreateConvolutionDescriptor(&plan->convolution),
             "cudnnCreateConvolutionDescriptor");
  checkCudnn(cudnnSetTensorNdDescriptor(plan->input, CUDNN_DATA_FLOAT, 4,
                                        inputDims.data(), inputStrides.data()),
             "cudnnSetTensorNdDescriptor(input)");
  checkCudnn(cudnnSetFilterNdDescriptor(plan->filter, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, 4,
                                        filterDims.data()),
             "cudnnSetFilterNdDescriptor");
  checkCudnn(cudnnSetTensorNdDescriptor(plan->output, CUDNN_DATA_FLOAT, 4,
                                        outputDims.data(),
                                        outputStrides.data()),
             "cudnnSetTensorNdDescriptor(output)");
  checkCudnn(cudnnSetConvolution2dDescriptor(
                 plan->convolution, 0, 0, strideH, strideW, dilationH,
                 dilationW, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
             "cudnnSetConvolution2dDescriptor");
  checkCudnn(cudnnSetConvolutionMathType(plan->convolution, CUDNN_FMA_MATH),
             "cudnnSetConvolutionMathType");

  int actualN, actualC, actualH, actualW;
  checkCudnn(cudnnGetConvolution2dForwardOutputDim(
                 plan->convolution, plan->input, plan->filter, &actualN,
                 &actualC, &actualH, &actualW),
             "cudnnGetConvolution2dForwardOutputDim");
  require(std::array<int, 4>{actualN, actualC, actualH, actualW} == outputDims,
          "output dimensions do not match convolution attributes");

  std::array<cudnnConvolutionFwdAlgoPerf_t, CUDNN_CONVOLUTION_FWD_ALGO_COUNT>
      results{};
  int returned = 0;
  checkCudnn(cudnnGetConvolutionForwardAlgorithm_v7(
                 getHandle(), plan->input, plan->filter, plan->convolution,
                 plan->output, static_cast<int>(results.size()), &returned,
                 results.data()),
             "cudnnGetConvolutionForwardAlgorithm_v7");
  bool found = false;
  for (int index = 0; index < returned; ++index) {
    if (results[index].status == CUDNN_STATUS_SUCCESS &&
        results[index].memory <= workspaceLimit()) {
      plan->algorithm = results[index].algo;
      plan->workspaceBytes = results[index].memory;
      found = true;
      break;
    }
  }
  require(found, "no cuDNN forward algorithm fits the workspace limit");
  if (plan->workspaceBytes)
    checkCuda(cudaMalloc(&plan->workspace, plan->workspaceBytes),
              "cudaMalloc(cuDNN workspace)");
  return plan;
}

} // namespace

#define TUTORIAL_MEMREF4_ARGS(name)                                            \
  float *name##Allocated, float *name##Aligned, int64_t name##Offset,          \
      int64_t name##Size0, int64_t name##Size1, int64_t name##Size2,           \
      int64_t name##Size3, int64_t name##Stride0, int64_t name##Stride1,       \
      int64_t name##Stride2, int64_t name##Stride3

extern "C" void tutorial_cudnn_conv2d_nchw_f32(
    TUTORIAL_MEMREF4_ARGS(input), TUTORIAL_MEMREF4_ARGS(filter),
    TUTORIAL_MEMREF4_ARGS(output), int64_t strideHValue, int64_t strideWValue,
    int64_t dilationHValue, int64_t dilationWValue) {
  (void)inputAllocated;
  (void)filterAllocated;
  (void)outputAllocated;
  require(inputSize0 == outputSize0, "batch dimensions do not match");
  require(inputSize1 == filterSize1 && filterSize0 == outputSize1,
          "convolution channel dimensions do not match");
  require(filterStride3 == 1 && filterStride2 == filterSize3 &&
              filterStride1 == filterSize2 * filterStride2 &&
              filterStride0 == filterSize1 * filterStride1,
          "cuDNN convolution requires a contiguous FCHW filter");

  std::array<int, 4> inputDims{checkedInt(inputSize0, "invalid input N"),
                               checkedInt(inputSize1, "invalid input C"),
                               checkedInt(inputSize2, "invalid input H"),
                               checkedInt(inputSize3, "invalid input W")};
  std::array<int, 4> inputStrides{
      checkedInt(inputStride0, "invalid input N stride"),
      checkedInt(inputStride1, "invalid input C stride"),
      checkedInt(inputStride2, "invalid input H stride"),
      checkedInt(inputStride3, "invalid input W stride")};
  std::array<int, 4> filterDims{checkedInt(filterSize0, "invalid filter F"),
                                checkedInt(filterSize1, "invalid filter C"),
                                checkedInt(filterSize2, "invalid filter H"),
                                checkedInt(filterSize3, "invalid filter W")};
  std::array<int, 4> outputDims{checkedInt(outputSize0, "invalid output N"),
                                checkedInt(outputSize1, "invalid output F"),
                                checkedInt(outputSize2, "invalid output H"),
                                checkedInt(outputSize3, "invalid output W")};
  std::array<int, 4> outputStrides{
      checkedInt(outputStride0, "invalid output N stride"),
      checkedInt(outputStride1, "invalid output F stride"),
      checkedInt(outputStride2, "invalid output H stride"),
      checkedInt(outputStride3, "invalid output W stride")};
  int strideH = checkedInt(strideHValue, "invalid convolution H stride");
  int strideW = checkedInt(strideWValue, "invalid convolution W stride");
  int dilationH = checkedInt(dilationHValue, "invalid convolution H dilation");
  int dilationW = checkedInt(dilationWValue, "invalid convolution W dilation");

  ConvKey key{};
  size_t keyIndex = 0;
  for (int value : inputDims)
    key[keyIndex++] = value;
  for (int value : inputStrides)
    key[keyIndex++] = value;
  for (int value : filterDims)
    key[keyIndex++] = value;
  for (int value : outputDims)
    key[keyIndex++] = value;
  for (int value : outputStrides)
    key[keyIndex++] = value;
  for (int value : {strideH, strideW, dilationH, dilationW})
    key[keyIndex++] = value;

  static thread_local std::unordered_map<ConvKey, std::unique_ptr<ConvPlan>,
                                         ConvKeyHash>
      plans;
  auto &plan = plans[key];
  if (!plan)
    plan = createPlan(inputDims, inputStrides, filterDims, outputDims,
                      outputStrides, strideH, strideW, dilationH, dilationW);

  const float alpha = 1.0f;
  const float beta = 1.0f;
  checkCudnn(cudnnConvolutionForward(
                 getHandle(), &alpha, plan->input, inputAligned + inputOffset,
                 plan->filter, filterAligned + filterOffset, plan->convolution,
                 plan->algorithm, plan->workspace, plan->workspaceBytes, &beta,
                 plan->output, outputAligned + outputOffset),
             "cudnnConvolutionForward");

  if (debugEnabled()) {
    static std::atomic<uint64_t> calls{0};
    uint64_t call = ++calls;
    std::fprintf(stderr,
                 "tutorial_gpu_runtime,op=cudnn_conv2d,call=%llu,algo=%d,"
                 "workspace_bytes=%zu,n=%d,c=%d,h=%d,w=%d,f=%d,kh=%d,kw=%d,"
                 "oh=%d,ow=%d\n",
                 static_cast<unsigned long long>(call),
                 static_cast<int>(plan->algorithm), plan->workspaceBytes,
                 inputDims[0], inputDims[1], inputDims[2], inputDims[3],
                 filterDims[0], filterDims[2], filterDims[3], outputDims[2],
                 outputDims[3]);
  }
}

#undef TUTORIAL_MEMREF4_ARGS
