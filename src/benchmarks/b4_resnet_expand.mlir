// ResNet-50 layer2 expansion (28x28=784 patches, 64->256 channels)
// Full matrices: 1.04MB (fills L2). Tile 32x64: 32KB (fits in L1).
func.func @b4_resnet_expand(%arg0: tensor<784x64xf32>, %arg1: tensor<64x256xf32>) -> tensor<784x256xf32> {
  %init = tensor.empty() : tensor<784x256xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<784x64xf32>, tensor<64x256xf32>)
                        outs(%init : tensor<784x256xf32>) -> tensor<784x256xf32>
  return %res : tensor<784x256xf32>
}
