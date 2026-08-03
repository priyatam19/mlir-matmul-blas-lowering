#map0 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @sample_model(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<3x5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x5xf32>) -> tensor<3x5xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<3x4xf32>, tensor<4x5xf32>) outs(%1 : tensor<3x5xf32>) -> tensor<3x5xf32>
    return %2 : tensor<3x5xf32>
  }
}
