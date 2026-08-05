// RUN: not tutorial-opt %s -lower-contraction-to-gpu="block-m=128 block-n=128 block-k=32 threads=256 stages=2" 2>&1 | FileCheck %s

module {}

// CHECK: GPU contraction configuration uses 65536 bytes of workgroup memory; limit is 49152
