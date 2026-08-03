// ResNet-50 bottleneck conv3a as GEMM (256 patches, 128->256 channels)
func.func @resnet_conv(%arg0: tensor<256x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<256x256xf32> {
  %init = tensor.empty() : tensor<256x256xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<256x128xf32>, tensor<128x256xf32>)
                        outs(%init : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %res : tensor<256x256xf32>
}
