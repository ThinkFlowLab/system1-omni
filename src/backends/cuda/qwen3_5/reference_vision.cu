#include "common.cuh"
#include "ops.h"
namespace cs1 { namespace reference_vision {
struct Moments { float m, v, n; };
__device__ Moments join_moments(Moments current, Moments other) {
    float count=current.n+other.n;
    if (count==0.f) return {0.f,0.f,0.f};
    float inv=1.f/count;
    float left=other.n*inv, right=current.n*inv;
    float delta=current.m-other.m;
    return {left*other.m+right*current.m,
            other.v+current.v+delta*delta*other.n*right,count};
}
__global__ void reference_norm_kernel(const bf16* x,const bf16* w,const bf16* b,bf16* y,int d) {
    __shared__ Moments warps[4];
    int tid=threadIdx.x, lane=tid%32, warp=tid/32;
    size_t base=(size_t)blockIdx.x*d;
    Moments a{0.f,0.f,0.f};
    for (int group=tid;group<d/4;group+=128) {
        #pragma unroll
        for (int j=0;j<4;j++) {
            float value=f32(x[base+group*4+j]);
            float delta=value-a.m;
            float count=a.n+1.f;
            float mean=a.m+delta*(1.f/count);
            a={mean,a.v+delta*(value-mean),count};
        }
    }
    for(int step=16;step>0;step/=2) {
        Moments other{__shfl_down_sync(0xffffffff,a.m,step),__shfl_down_sync(0xffffffff,a.v,step),__shfl_down_sync(0xffffffff,a.n,step)};
        a=join_moments(a,other);
    }
    if(lane==0) warps[warp]=a;
    __syncthreads();
    for(int step=2;step>0;step/=2) {
        if(lane==0&&warp<step) a=join_moments(a,warps[warp+step]);
        __syncthreads();
        if(lane==0&&warp<step) warps[warp]=a;
        __syncthreads();
    }
    float mean=warps[0].m, inv=rsqrtf(warps[0].v/(float)d+1e-6f);
    for(int group=tid;group<d/4;group+=128) {
        #pragma unroll
        for(int j=0;j<4;j++) {
            int c=group*4+j;
            y[base+c]=to_bf16(f32(w[c])*(inv*(f32(x[base+c])-mean))+f32(b[c]));
        }
    }
}

__global__ void rope_v2_kernel(const bf16* qkv,const float* co,const float* si,bf16* q,bf16* k,size_t n,int hidden,int dh) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int token=i/hidden,d=i%dh,channel=i%hidden,half=dh/2;
    const float angle=co[(size_t)token*half+d%half];
    const float c=cosf(angle),s=sinf(angle);
    const int partner=channel+(d<half?half:-half);
    const float sign=d<half?-1.f:1.f;
    q[i]=to_bf16(__fadd_rn(__fmul_rn(f32(qkv[(size_t)token*hidden*3+channel]),c),__fmul_rn(sign*f32(qkv[(size_t)token*hidden*3+partner]),s)));
    k[i]=to_bf16(__fadd_rn(__fmul_rn(f32(qkv[(size_t)token*hidden*3+hidden+channel]),c),__fmul_rn(sign*f32(qkv[(size_t)token*hidden*3+hidden+partner]),s)));
}
__global__ void pad_attention_kernel(const bf16* q,const bf16* k,const bf16* v,bf16* pq,bf16* pk,bf16* pv,size_t size,int heads,int dh) {
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=size) return;
    const int d=i%80,head=(i/80)%heads;
    const size_t token=i/(heads*80),compact=(token*heads+head)*dh+d;
    pq[i]=d<dh?q[compact]:to_bf16(0.f);
    pk[i]=d<dh?k[compact]:to_bf16(0.f);
    pv[i]=d<dh?v[token*heads*dh*3+head*dh+d]:to_bf16(0.f);
}
__global__ void unpack_attention_kernel(const bf16* padded,bf16* out,size_t size,int dh) {
    const size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size) out[i]=padded[(i/dh)*80+i%dh];
}
} }
using namespace cs1;
extern "C" int cs1_reference_vision_norm(const void* x,const void* w,const void* b,void* y,int rows,int d,void* stream) {
    if(rows<=0 || d!=1152) return cudaErrorInvalidValue;
    reference_vision::reference_norm_kernel<<<rows,128,0,(cudaStream_t)stream>>>((const bf16*)x,(const bf16*)w,(const bf16*)b,(bf16*)y,d);
    return cudaGetLastError();
}
extern "C" int cs1_reference_vision_rope(const void* qkv,const float* co,const float* si,void* q,void* k,int n,int hidden,int dh,void* stream) {
    if(n<=0 || (hidden!=1152 || dh!=72)) return cudaErrorInvalidValue;
    const size_t count=(size_t)n*hidden;
    reference_vision::rope_v2_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((const bf16*)qkv,co,si,(bf16*)q,(bf16*)k,count,hidden,dh);
    return cudaGetLastError();
}
extern "C" int cs1_reference_vision_attention(const void* q,const void* k,const void* v,void* out,int n,int heads,int dh,void* workspace,void* stream) {
    if(n<=0 || heads!=16 || dh!=72) return cudaErrorInvalidValue;
    if(!workspace) return cudaErrorInvalidValue;
    const size_t count=(size_t)n*heads*80;
    bf16* pq=(bf16*)workspace; bf16* pk=pq+count; bf16* pv=pk+count; bf16* po=pv+count;
    reference_vision::pad_attention_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)v,pq,pk,pv,count,heads,dh);
    cudaError_t status=cudaGetLastError(); if(status!=cudaSuccess) return status;
    status=(cudaError_t)cs1_reference_vision_attention_padded(pq,pk,pv,po,n,heads,po+count,stream);
    if(status!=cudaSuccess) return status;
    status=cudaGetLastError(); if(status!=cudaSuccess) return status;
    reference_vision::unpack_attention_kernel<<<((size_t)n*heads*dh+255)/256,256,0,(cudaStream_t)stream>>>(po,(bf16*)out,(size_t)n*heads*dh,dh);
    return cudaGetLastError();
}
