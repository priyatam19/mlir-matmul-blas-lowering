// RUN: not tutorial-opt %s -tile-batch-matmul-for-gpu="block-m=0 block-n=32" 2>&1 | FileCheck %s --check-prefix=NONPOSITIVE
// RUN: not tutorial-opt %s -tile-batch-matmul-for-gpu="block-m=64 block-n=32" 2>&1 | FileCheck %s --check-prefix=TOO-MANY

module {
}

// NONPOSITIVE: error: GPU block dimensions must be positive and contain at most 1024 threads, got block-m=0 block-n=32
// TOO-MANY: error: GPU block dimensions must be positive and contain at most 1024 threads, got block-m=64 block-n=32
