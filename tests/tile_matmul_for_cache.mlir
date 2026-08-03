// RUN: tutorial-opt <%s -tile-matmul-for-cache="tile-m=2 tile-n=3 tile-k=4" -split-input-file | FileCheck %s

func.func @tile_matmul(%lhs: tensor<8x12xf32>, %rhs: tensor<12x9xf32>, %acc: tensor<8x9xf32>) -> tensor<8x9xf32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<8x12xf32>, tensor<12x9xf32>)
    outs(%acc: tensor<8x9xf32>)
  -> tensor<8x9xf32>
  return %result: tensor<8x9xf32>
}

// CHECK-LABEL: func.func @tile_matmul
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract_slice
// CHECK: linalg.matmul
// CHECK: tensor.insert_slice
