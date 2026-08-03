// Long-sequence attention score (seq=512, head_dim=64)
// Full matrices: 1.28MB (spills L2=1.25MB). Tile 64x64: 48KB (≈ L1 boundary).
func.func @b3_long_seq_attn(%arg0: tensor<512x64xf32>, %arg1: tensor<64x512xf32>) -> tensor<512x512xf32> {
  %init = tensor.empty() : tensor<512x512xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<512x64xf32>, tensor<64x512xf32>)
                        outs(%init : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %res : tensor<512x512xf32>
}
