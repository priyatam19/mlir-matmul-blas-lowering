// RUN: not tutorial-opt %s -tile-conv2d-nchw-for-gpu="threads=0" 2>&1 | FileCheck %s --check-prefix=ZERO
// RUN: not tutorial-opt %s -tile-conv2d-nchw-for-gpu="threads=1025" 2>&1 | FileCheck %s --check-prefix=TOO-MANY

module {
}

// ZERO: error: GPU convolution thread count must be between 1 and 1024, got 0
// TOO-MANY: error: GPU convolution thread count must be between 1 and 1024, got 1025
