#include "common.cuh"
using namespace cs1;
template<bool Add>
__global__ void reference_norm(const bf16* x,const bf16* delta,const bf16* w,bf16* out,int D,float eps) {
 __shared__ float shared[512];
 int lane=threadIdx.x;size_t row=blockIdx.x;x+=row*D;if(delta)delta+=row*D;out+=row*D;float sum[4]={};
 for(int base=lane*4;base<D;base+=blockDim.x*4) {
#pragma unroll
  for(int j=0;j<4;j++) {float v=f32(x[base+j]);if constexpr(Add){v=round_bf16(v+f32(delta[base+j]));const_cast<bf16*>(x)[base+j]=to_bf16(v);}sum[j]=__fadd_rn(sum[j],__fmul_rn(v,v));}
 }
 float value=__fadd_rn(__fadd_rn(__fadd_rn(sum[0],sum[1]),sum[2]),sum[3]);
 shared[lane]=value;
 for(int step=blockDim.x/2;step>=32;step/=2) {__syncthreads();if(lane<step) {value=__fadd_rn(value,shared[lane+step]);shared[lane]=value;}}
 if(lane<32){for(int step=16;step>0;step/=2)value=__fadd_rn(value,__shfl_down_sync(0xffffffff,value,step));if(lane==0)shared[0]=rsqrtf(__fadd_rn(__fmul_rn(value,1.f/D),eps));}
 __syncthreads();float inverse=shared[0];
 for(int i=lane;i<D;i+=blockDim.x)out[i]=to_bf16(__fmul_rn(__fmul_rn(f32(x[i]),inverse),__fadd_rn(1.f,f32(w[i]))));
}
extern "C" int reference_norm_run(const void* x,const void* delta,const void* w,void* out,int rows,int D,float eps,void* stream) {
 int powRows=1,powD=1;while(powRows*2<=rows&&powRows<512)powRows*=2;while(powD*2<=D/4&&powD<512)powD*=2;
 int width=min(powD,32),height=min(powRows,512/width);width=min(powD,512/height);
 int threads=width;if((D+width-1)/width>=min(height*16,256))threads*=height;
 if(delta)reference_norm<true><<<rows,threads,0,(cudaStream_t)stream>>>((const bf16*)x,(const bf16*)delta,(const bf16*)w,(bf16*)out,D,eps);
 else reference_norm<false><<<rows,threads,0,(cudaStream_t)stream>>>((const bf16*)x,nullptr,(const bf16*)w,(bf16*)out,D,eps);
 return cudaGetLastError();
}

extern "C" int cs1_reference_rms_norm(const void* x,const void* w,void* out,int rows,int d,float eps,void* stream) {
    if(rows<=0 || d!=5120) return cudaErrorInvalidValue;
    return reference_norm_run(x,nullptr,w,out,rows,d,eps,stream);
}
extern "C" int cs1_reference_add_rms_norm(void* x,const void* delta,const void* w,void* out,int rows,int d,float eps,void* stream) {
    if(rows<=0 || d!=5120) return cudaErrorInvalidValue;
    return reference_norm_run(x,delta,w,out,rows,d,eps,stream);
}
