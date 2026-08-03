// Benchmark harness: gpt2_ffn  (128x256) x (256x512) -> (128x512)
// RUNS=300
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

template <typename T, int NDIM> struct MemRefDescriptor {
  T *allocated; T *aligned; int64_t offset; int64_t sizes[NDIM]; int64_t strides[NDIM];
};

extern "C" {
void _mlir_ciface_gpt2_ffn(
    MemRefDescriptor<float,2> *result,
    MemRefDescriptor<float,2> *A,
    MemRefDescriptor<float,2> *B);
}

static float A_data[128*256];
static float B_data[256*512];
// C_data only for initializing the descriptor; MLIR allocates its own output buffer
static float C_data[128*512];

int main() {
  for (int i = 0; i < 128*256; i++) A_data[i] = 0.001f*(float)(i%100+1);
  for (int i = 0; i < 256*512; i++) B_data[i] = 0.001f*(float)((i+3)%100+1);

  MemRefDescriptor<float,2> A = {A_data, A_data, 0, {128,256}, {256,1}};
  MemRefDescriptor<float,2> B = {B_data, B_data, 0, {256,512}, {512,1}};

  // Warm-up: 30 calls to stabilize caches and branch predictors
  for (int i = 0; i < 30; i++) {
    MemRefDescriptor<float,2> C = {C_data, C_data, 0, {128,512}, {512,1}};
    _mlir_ciface_gpt2_ffn(&C, &A, &B);
  }

  std::vector<long long> times;
  times.reserve(300);
  for (int i = 0; i < 300; i++) {
    MemRefDescriptor<float,2> C = {C_data, C_data, 0, {128,512}, {512,1}};
    auto t0 = std::chrono::high_resolution_clock::now();
    _mlir_ciface_gpt2_ffn(&C, &A, &B);
    auto t1 = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
  }

  std::sort(times.begin(), times.end());
  long long p50 = times[300/2];
  long long p10 = times[300/10];
  long long p90 = times[300*9/10];
  long long sum = 0; for (auto t:times) sum+=t;
  long long avg = sum/300;
  printf("avg=%lld ns  p50=%lld ns  p10=%lld ns  p90=%lld ns\n",avg,p50,p10,p90);
  return 0;
}
