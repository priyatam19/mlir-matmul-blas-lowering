// GPT-2 small FFN first linear (batch=128 tokens, d_model=256->d_ffn=512)
func.func @gpt2_ffn(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %init = tensor.empty() : tensor<128x512xf32>
  %res  = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
                        outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %res : tensor<128x512xf32>
}
