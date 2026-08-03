#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void _mlir_ciface_sample_model(MemRefDescriptor<float, 2> *output,
                               MemRefDescriptor<float, 2> *input,
                               MemRefDescriptor<float, 2> *input_shift,
                               MemRefDescriptor<float, 2> *weight_t,
                               MemRefDescriptor<float, 1> *bias);
}

static MemRefDescriptor<float, 2> make2D(float *data, int64_t rows,
                                         int64_t cols) {
  return {data, data, 0, {rows, cols}, {cols, 1}};
}

static MemRefDescriptor<float, 1> make1D(float *data, int64_t size) {
  return {data, data, 0, {size}, {1}};
}

int main() {
  float input[12];
  float inputShift[12] = {
      0.236446381f, 0.226617992f, 0.800530195f, 0.169187665f,
      0.264958382f, 0.771991789f, 0.128188312f, 0.745214045f,
      0.804467618f, 0.635709107f, 0.589616418f, 0.693289578f};
  float weightT[20] = {
      0.37824893f,   0.366648793f,  -0.441477418f, 0.188260794f,
      -0.32082057f,  0.0406756401f, -0.011584878f, 0.33135587f,
      -0.0991921425f, -0.111118853f, -0.359977543f, -0.292298317f,
      -0.0433897376f, -0.319720566f, 0.297240615f, 0.461343944f,
      -0.193659544f, 0.344493091f,  0.174016714f,  -0.272269189f};
  float bias[5] = {-0.0244604945f, -0.067863822f, 0.0572949648f,
                   0.481411994f, 0.292283773f};

  for (float &v : input)
    v = 1.0f;

  auto inputRef = make2D(input, 3, 4);
  auto shiftRef = make2D(inputShift, 3, 4);
  auto weightRef = make2D(weightT, 4, 5);
  auto biasRef = make1D(bias, 5);
  MemRefDescriptor<float, 2> outputRef = {};

  constexpr int warmups = 20;
  constexpr int runs = 1000;

  for (int i = 0; i < warmups; ++i) {
    _mlir_ciface_sample_model(&outputRef, &inputRef, &shiftRef, &weightRef,
                              &biasRef);
    std::free(outputRef.allocated);
    outputRef = {};
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < runs; ++i) {
    _mlir_ciface_sample_model(&outputRef, &inputRef, &shiftRef, &weightRef,
                              &biasRef);
    std::free(outputRef.allocated);
    outputRef = {};
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "CPU avg inference time: " << std::fixed
            << std::setprecision(9) << (elapsed.count() / runs) << " sec\n";

  _mlir_ciface_sample_model(&outputRef, &inputRef, &shiftRef, &weightRef,
                            &biasRef);
  for (int64_t i = 0; i < outputRef.sizes[0]; ++i) {
    for (int64_t j = 0; j < outputRef.sizes[1]; ++j)
      std::cout << std::fixed << std::setprecision(5)
                << outputRef.aligned[i * outputRef.strides[0] + j] << ' ';
    std::cout << "\n";
  }
  std::free(outputRef.allocated);
  return 0;
}
