func.func @gpu_bmm_long(%a: memref<12x512x64xf32>,
                        %b: memref<12x64x512xf32>,
                        %c: memref<12x512x512xf32>) {
  linalg.batch_matmul
      ins(%a, %b : memref<12x512x64xf32>, memref<12x64x512xf32>)
      outs(%c : memref<12x512x512xf32>)
  return
}
