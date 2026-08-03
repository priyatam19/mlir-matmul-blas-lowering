// BERT-base: Q @ K^T attention score (seq=128, head_dim=64)
// Full matrices: 160KB (exceeds L1=48KB). Tile 32x32: 20KB (fits in L1).
func.func @b1_bert_attn(%arg0: tensor<128x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<128x128xf32> {
  %init = tensor.empty() : tensor<128x128xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x128xf32>)
                        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}
