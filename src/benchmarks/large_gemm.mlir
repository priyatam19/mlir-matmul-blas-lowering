// Transformer QKV projection: batch=512, hidden=256, out=512
func.func @large_gemm(%arg0: tensor<512x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<512x512xf32> {
  %init = tensor.empty() : tensor<512x512xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<512x256xf32>, tensor<256x512xf32>)
                        outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %res : tensor<512x512xf32>
}
