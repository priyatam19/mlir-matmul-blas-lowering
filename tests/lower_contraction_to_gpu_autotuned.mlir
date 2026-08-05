// RUN: tutorial-opt %s -lower-contraction-to-gpu="strategy=autotuned target=sm_89" | FileCheck %s

func.func @autotuned_matmul(%lhs: memref<?x?xf32>,
                            %rhs: memref<?x?xf32>,
                            %out: memref<?x?xf32>) {
  linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out : memref<?x?xf32>)
  return
}

// CHECK-DAG: func.func private @tutorial_autotune_begin(i64, i64) -> index
// CHECK-DAG: func.func private @tutorial_autotune_end(i64, index)
// CHECK-LABEL: func.func @autotuned_matmul
// CHECK: %[[CANDIDATE:.*]] = call @tutorial_autotune_begin
// CHECK: scf.index_switch %[[CANDIDATE]]
// CHECK: case 0 {
// CHECK: scf.parallel
// CHECK: case 1 {
// CHECK: gpu.launch
// CHECK: case 6 {
// CHECK: gpu.launch
// CHECK: default {
// CHECK: gpu.launch
// CHECK: call @tutorial_autotune_end
// CHECK-NOT: linalg.matmul
