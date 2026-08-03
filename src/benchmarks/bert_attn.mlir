// BERT-base: single attention-head QK^T  (seq_len=128, head_dim=64)
// No linalg.fill: cblas_sgemm with beta=0 overwrites C entirely.
func.func @bert_attn(%arg0: tensor<128x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<128x128xf32> {
  %init = tensor.empty() : tensor<128x128xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<128x64xf32>, tensor<64x128xf32>)
                        outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}
