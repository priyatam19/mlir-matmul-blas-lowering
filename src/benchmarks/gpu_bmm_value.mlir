func.func @gpu_bmm_value(%a: memref<32x128x64xf32>,
                         %b: memref<32x64x64xf32>,
                         %c: memref<32x128x64xf32>) {
  linalg.batch_matmul
      ins(%a, %b : memref<32x128x64xf32>, memref<32x64x64xf32>)
      outs(%c : memref<32x128x64xf32>)
  return
}
