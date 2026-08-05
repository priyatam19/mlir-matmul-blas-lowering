#ifndef MLIR_TUTORIAL_GPU_RUNTIME_H
#define MLIR_TUTORIAL_GPU_RUNTIME_H

#include <cstdint>

#define TUTORIAL_MEMREF3_ARGS(name)                                            \
  float *name##Allocated, float *name##Aligned, int64_t name##Offset,          \
      int64_t name##Size0, int64_t name##Size1, int64_t name##Size2,           \
      int64_t name##Stride0, int64_t name##Stride1, int64_t name##Stride2

extern "C" void
    tutorial_cublas_sgemm_strided_batched_f32(TUTORIAL_MEMREF3_ARGS(lhs),
                                              TUTORIAL_MEMREF3_ARGS(rhs),
                                              TUTORIAL_MEMREF3_ARGS(output));

#undef TUTORIAL_MEMREF3_ARGS

#endif // MLIR_TUTORIAL_GPU_RUNTIME_H
