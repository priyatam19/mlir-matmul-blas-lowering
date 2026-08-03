// Benchmark: b2_mobilenet_conv  (3136x32) x (32x64) -> (3136x64)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
template <typename T, int NDIM> struct MemRefDescriptor {
  T *allocated; T *aligned; int64_t offset; int64_t sizes[NDIM]; int64_t strides[NDIM];
};
extern "C" {
void _mlir_ciface_b2_mobilenet_conv(
    MemRefDescriptor<float,2> *result,
    MemRefDescriptor<float,2> *A,
    MemRefDescriptor<float,2> *B);
}
static float A_data[3136*32], B_data[32*64], C_data[3136*64];
int main() {
  for (int i=0;i<3136*32;i++) A_data[i]=0.001f*(float)(i%100+1);
  for (int i=0;i<32*64;i++) B_data[i]=0.001f*(float)((i+3)%100+1);
  MemRefDescriptor<float,2> A={A_data,A_data,0,{3136,32},{32,1}};
  MemRefDescriptor<float,2> B={B_data,B_data,0,{32,64},{64,1}};
  for (int i=0;i<30;i++) {
    MemRefDescriptor<float,2> C={C_data,C_data,0,{3136,64},{64,1}};
    _mlir_ciface_b2_mobilenet_conv(&C,&A,&B);
  }
  std::vector<long long> times; times.reserve(200);
  for (int i=0;i<200;i++) {
    MemRefDescriptor<float,2> C={C_data,C_data,0,{3136,64},{64,1}};
    auto t0=std::chrono::high_resolution_clock::now();
    _mlir_ciface_b2_mobilenet_conv(&C,&A,&B);
    auto t1=std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count());
  }
  std::sort(times.begin(),times.end());
  long long p50=times[200/2],p10=times[200/10],p90=times[200*9/10],sum=0;
  for (auto t:times) sum+=t;
  printf("avg=%lld ns  p50=%lld ns  p10=%lld ns  p90=%lld ns\n",sum/200,p50,p10,p90);
  return 0;
}
