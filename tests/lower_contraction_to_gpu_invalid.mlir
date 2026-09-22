// RUN: not tutorial-opt %s -lower-contraction-to-gpu="block-m=128 block-n=128 block-k=32 threads=256 stages=2" 2>&1 | FileCheck %s
// RUN: not tutorial-opt %s -lower-contraction-to-gpu="strategy=shared-fp32 block-m=128 block-n=256 block-k=16 threads=128 stages=1" 2>&1 | FileCheck %s --check-prefix=REGISTERS
// RUN: not tutorial-opt %s -lower-contraction-to-gpu="strategy=autotuned autotune-math-mode=fast-ish" 2>&1 | FileCheck %s --check-prefix=MATH

module {}

// CHECK: GPU contraction configuration uses 65536 bytes of workgroup memory; limit is 49152
// REGISTERS: GPU contraction configuration is estimated to use 328 registers per thread; limit is 192
// MATH: unknown autotune math mode: fast-ish
