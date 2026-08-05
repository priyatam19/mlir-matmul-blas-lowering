func.func @gpu_bmm_irregular(%a: memref<7x129x65xf32>,
                             %b: memref<7x65x127xf32>,
                             %c: memref<7x129x127xf32>) {
  linalg.batch_matmul
      ins(%a, %b : memref<7x129x65xf32>, memref<7x65x127xf32>)
      outs(%c : memref<7x129x127xf32>)
  return
}
