#ifndef MLIR_TUTORIAL_AUTOTUNE_RUNTIME_H
#define MLIR_TUTORIAL_AUTOTUNE_RUNTIME_H

#include <cstdint>

extern "C" int64_t tutorial_autotune_begin(int64_t key,
                                            int64_t candidateCount);
extern "C" void tutorial_autotune_end(int64_t key, int64_t candidate);

#endif // MLIR_TUTORIAL_AUTOTUNE_RUNTIME_H
