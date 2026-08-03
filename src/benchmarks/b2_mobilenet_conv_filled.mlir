// MobileNet V1 pointwise conv3 with explicit zero initialization.
// Shape: (3136x32) x (32x64) -> (3136x64)
func.func @b2_mobilenet_conv(%arg0: tensor<3136x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<3136x64xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<3136x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<3136x64xf32>) -> tensor<3136x64xf32>
  %res = linalg.matmul ins(%arg0, %arg1 : tensor<3136x32xf32>, tensor<32x64xf32>)
                       outs(%init : tensor<3136x64xf32>) -> tensor<3136x64xf32>
  return %res : tensor<3136x64xf32>
}
