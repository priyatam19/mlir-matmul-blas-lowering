func.func @gpu_bmm_bert(%a: memref<12x128x64xf32>,
                        %b: memref<12x64x128xf32>,
                        %c: memref<12x128x128xf32>) {
  linalg.batch_matmul
      ins(%a, %b : memref<12x128x64xf32>, memref<12x64x128xf32>)
      outs(%c : memref<12x128x128xf32>)
  return
}
