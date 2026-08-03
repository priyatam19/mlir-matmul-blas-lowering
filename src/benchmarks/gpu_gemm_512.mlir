func.func @gpu_gemm_512(%a: memref<512x256xf32>,
                        %b: memref<256x512xf32>,
                        %c: memref<512x512xf32>) {
  linalg.matmul ins(%a, %b : memref<512x256xf32>, memref<256x512xf32>)
                outs(%c : memref<512x512xf32>)
  return
}
