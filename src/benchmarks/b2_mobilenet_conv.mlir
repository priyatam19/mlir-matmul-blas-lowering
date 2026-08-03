// MobileNet V1 pointwise conv3 (56x56=3136 patches, 32 input -> 64 output channels)
// Full matrices: 1.19MB (fills L2). Tile 64x64: 32KB (fits in L1).
func.func @b2_mobilenet_conv(%arg0: tensor<3136x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<3136x64xf32> {
  %init = tensor.empty() : tensor<3136x64xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<3136x32xf32>, tensor<32x64xf32>)
                        outs(%init : tensor<3136x64xf32>) -> tensor<3136x64xf32>
  return %res : tensor<3136x64xf32>
}
