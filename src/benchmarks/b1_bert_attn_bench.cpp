// Benchmark: b1_bert_attn  (128x64) x (64x128) -> (128x128)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
template <typename T, int NDIM> struct MemRefDescriptor {
  T *allocated; T *aligned; int64_t offset; int64_t sizes[NDIM]; int64_t strides[NDIM];
};
extern "C" {
void _mlir_ciface_b1_bert_attn(
    MemRefDescriptor<float,2> *result,
    MemRefDescriptor<float,2> *A,
    MemRefDescriptor<float,2> *B);
}
static float A_data[128*64], B_data[64*128], C_data[128*128];
int main() {
  for (int i=0;i<128*64;i++) A_data[i]=0.001f*(float)(i%100+1);
  for (int i=0;i<64*128;i++) B_data[i]=0.001f*(float)((i+3)%100+1);
  MemRefDescriptor<float,2> A={A_data,A_data,0,{128,64},{64,1}};
  MemRefDescriptor<float,2> B={B_data,B_data,0,{64,128},{128,1}};
  for (int i=0;i<30;i++) {
    MemRefDescriptor<float,2> C={C_data,C_data,0,{128,128},{128,1}};
    _mlir_ciface_b1_bert_attn(&C,&A,&B);
  }
  std::vector<long long> times; times.reserve(500);
  for (int i=0;i<500;i++) {
    MemRefDescriptor<float,2> C={C_data,C_data,0,{128,128},{128,1}};
    auto t0=std::chrono::high_resolution_clock::now();
    _mlir_ciface_b1_bert_attn(&C,&A,&B);
    auto t1=std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
  }
  std::sort(times.begin(),times.end());
  long long p50=times[500/2],p10=times[500/10],p90=times[500*9/10],sum=0;
  for (auto t:times) sum+=t;
  printf("avg=%lld ns  p50=%lld ns  p10=%lld ns  p90=%lld ns\n",sum/500,p50,p10,p90);
  return 0;
}
