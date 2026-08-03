#ifndef MLIR_TUTORIAL_GPU_RUNTIME_H
#define MLIR_TUTORIAL_GPU_RUNTIME_H

#include <cstdint>

#define TUTORIAL_MEMREF3_ARGS(name)                                            \
  float *name##Allocated, float *name##Aligned, int64_t name##Offset,          \
      int64_t name##Size0, int64_t name##Size1, int64_t name##Size2,           \
      int64_t name##Stride0, int64_t name##Stride1, int64_t name##Stride2

#define TUTORIAL_MEMREF4_ARGS(name)                                            \
  float *name##Allocated, float *name##Aligned, int64_t name##Offset,          \
      int64_t name##Size0, int64_t name##Size1, int64_t name##Size2,           \
      int64_t name##Size3, int64_t name##Stride0, int64_t name##Stride1,       \
      int64_t name##Stride2, int64_t name##Stride3

extern "C" void
    tutorial_cublas_sgemm_strided_batched_f32(TUTORIAL_MEMREF3_ARGS(lhs),
                                              TUTORIAL_MEMREF3_ARGS(rhs),
                                              TUTORIAL_MEMREF3_ARGS(output));

extern "C" void tutorial_cudnn_conv2d_nchw_f32(TUTORIAL_MEMREF4_ARGS(input),
                                               TUTORIAL_MEMREF4_ARGS(filter),
                                               TUTORIAL_MEMREF4_ARGS(output),
                                               int64_t strideH, int64_t strideW,
                                               int64_t dilationH,
                                               int64_t dilationW);

#undef TUTORIAL_MEMREF3_ARGS
#undef TUTORIAL_MEMREF4_ARGS

#endif // MLIR_TUTORIAL_GPU_RUNTIME_H
