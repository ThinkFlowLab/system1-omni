// Native Qwen3.5 vision kernels. BF16 activations, FP32 normalization/rotary/LoRA.
#include "common.cuh"
#include "mma.cuh"
#include "ops.h"
namespace cs1 { namespace vision {
namespace flash {

constexpr int BM = 64, BN = 32, THREADS = 128;
template<int D> constexpr int SMEM_BYTES = (BM + 2 * BN) * (D + 8) * 2;

template<int D>
__global__ void __launch_bounds__(THREADS)
    flash_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, int ldv,
                 bf16* __restrict__ out, int T, int Hq, int Hk, float scale_log2) {
    constexpr int LDS = D + 8;
    extern __shared__ __align__(16) unsigned char smem[];
    bf16* qs = reinterpret_cast<bf16*>(smem);
    bf16* ks = qs + BM * LDS;
    bf16* vs = ks + BN * LDS;
    const int h = blockIdx.y, hk = h / (Hq / Hk);
    const int q0 = (gridDim.x - 1 - blockIdx.x) * BM;  // the longest blocks first
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, t = lane % 4;
    const int row0 = q0 + warp * 16;  // this warp's first query

    for (int c = tid; c < BM * (D / 8); c += THREADS) {
        const int r = c / (D / 8), col = (c % (D / 8)) * 8, row = q0 + r;
        cp_async16(qs + r * LDS + col, q + ((size_t)min(row, T - 1) * Hq + h) * D + col, row < T);
    }
    cp_async_commit();

    float o[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    const int kv_end = T;
    for (int k0 = 0; k0 < kv_end; k0 += BN) {
        for (int c = tid; c < BN * (D / 8); c += THREADS) {
            const int r = c / (D / 8), col = (c % (D / 8)) * 8, s = k0 + r;
            cp_async16(ks + r * LDS + col, k + ((size_t)min(s, T - 1) * Hk + hk) * D + col, s < T);
        }
        cp_async_commit();
        for (int c = tid; c < BN * (D / 8); c += THREADS) {
            const int r = c / (D / 8), col = (c % (D / 8)) * 8, s = k0 + r;
            cp_async16(vs + r * LDS + col, v + (size_t)min(s, T - 1) * ldv + (size_t)hk * D + col, s < T);
        }
        cp_async_commit();
        cp_async_wait<1>();  // Q and K
        __syncthreads();

        // keys past every query of this warp contribute nothing
        const bool active = true;
        float sc[BN / 8][4];
#pragma unroll
        for (int n = 0; n < BN / 8; n++) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
        if (active) {
#pragma unroll
            for (int kk = 0; kk < D; kk += 16) {
                uint32_t a[4];
                load_a(a, qs, LDS, warp * 16, kk, lane);
#pragma unroll
                for (int n = 0; n < BN / 8; n += 2) {
                    uint32_t b[4];
                    load_b_nk(b, ks, LDS, kk, n * 8, lane);
                    mma16816(sc[n], a, b[0], b[1]);
                    mma16816(sc[n + 1], a, b[2], b[3]);
                }
            }
        }
        uint32_t p[BN / 16][4];
        if (active) {
            // causal and length mask, then the online softmax in base 2
            float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
            for (int n = 0; n < BN / 8; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int key = k0 + n * 8 + 2 * t + (e & 1);
                    sc[n][e] = (key < T) ? sc[n][e] * scale_log2 : -INFINITY;
                    mx[e >> 1] = fmaxf(mx[e >> 1], sc[n][e]);
                }
            }
            float alpha[2], base[2];
#pragma unroll
            for (int r = 0; r < 2; r++) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
                const float mn = fmaxf(m[r], mx[r]);
                base[r] = mn == -INFINITY ? 0.f : mn;
                alpha[r] = exp2f(m[r] - base[r]);
                m[r] = mn;
                l[r] *= alpha[r];
            }
#pragma unroll
            for (int n = 0; n < BN / 8; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    sc[n][e] = exp2f(sc[n][e] - base[e >> 1]);
                    l[e >> 1] += sc[n][e];
                }
            }
#pragma unroll
            for (int n = 0; n < D / 8; n++) {
                o[n][0] *= alpha[0];
                o[n][1] *= alpha[0];
                o[n][2] *= alpha[1];
                o[n][3] *= alpha[1];
            }
            // the score accumulators, two 8-key tiles at a time, are the A fragments of P*V
#pragma unroll
            for (int j = 0; j < BN / 16; j++) {
                p[j][0] = pack_bf16(sc[2 * j][0], sc[2 * j][1]);
                p[j][1] = pack_bf16(sc[2 * j][2], sc[2 * j][3]);
                p[j][2] = pack_bf16(sc[2 * j + 1][0], sc[2 * j + 1][1]);
                p[j][3] = pack_bf16(sc[2 * j + 1][2], sc[2 * j + 1][3]);
            }
        }
        cp_async_wait<0>();  // V
        __syncthreads();
        if (active) {
#pragma unroll
            for (int j = 0; j < BN / 16; j++) {
#pragma unroll
                for (int n = 0; n < D / 8; n += 2) {
                    uint32_t b[4];
                    load_b_kn(b, vs, LDS, j * 16, n * 8, lane);
                    mma16816(o[n], p[j], b[0], b[1]);
                    mma16816(o[n + 1], p[j], b[2], b[3]);
                }
            }
        }
        __syncthreads();  // before the next tile overwrites K and V
    }

    // the four lanes of a row each summed a quarter of its keys
#pragma unroll
    for (int r = 0; r < 2; r++) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
    const float inv[2] = {1.f / l[0], 1.f / l[1]};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = row0 + g + r * 8;
        if (row >= T) continue;
        bf16* dst = out + ((size_t)row * Hq + h) * D + 2 * t;
#pragma unroll
        for (int n = 0; n < D / 8; n++)
            *reinterpret_cast<uint32_t*>(dst + n * 8) = pack_bf16(o[n][2 * r] * inv[r], o[n][2 * r + 1] * inv[r]);
    }
}

}  // namespace flash
} }
namespace cs1 { namespace vision {
__global__ void norm_kernel(const bf16* x, const bf16* w, const bf16* b, bf16* y, int d) {
    __shared__ float scratch[32];
    const size_t off = (size_t)blockIdx.x*d;
    float sum = 0.f;
    for (int i=threadIdx.x;i<d;i+=blockDim.x) sum += f32(x[off+i]);
    const float mean=block_sum(sum,scratch)/d;
    float var=0.f;
    for (int i=threadIdx.x;i<d;i+=blockDim.x) { float a=f32(x[off+i])-mean; var+=a*a; }
    const float inv=rsqrtf(block_sum(var,scratch)/d+1e-6f);
    for (int i=threadIdx.x;i<d;i+=blockDim.x)
        y[off+i]=to_bf16((f32(x[off+i])-mean)*inv*f32(w[i])+f32(b[i]));
}
__global__ void position_kernel(bf16* x,const bf16* table,const int* indices,const float* weights,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int t=i/1024,d=i%1024;
    float pos=0.f;
    // Separate FP32 multiply and sum, then position BF16 rounding before the residual add.
    for(int j=0;j<4;j++) pos=__fadd_rn(pos,__fmul_rn(f32(table[indices[t*4+j]*1024+d]),weights[t*4+j]));
    x[i]=to_bf16(f32(x[i])+round_bf16(pos));
}
__global__ void rope_kernel(const bf16* qkv,const float* co,const float* si,bf16* q,bf16* k,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int t=i/1024,d=i%64,channel=i%1024;
    const float c=co[t*32+d%32],s=si[t*32+d%32];
    int partner=channel+(d<32?32:-32);
    float sign=d<32?-1.f:1.f;
    // PyTorch materializes both products in float32 (not a fused multiply-add).
    q[i]=to_bf16(__fadd_rn(__fmul_rn(f32(qkv[t*3072+channel]),c),__fmul_rn(sign*f32(qkv[t*3072+partner]),s)));
    k[i]=to_bf16(__fadd_rn(__fmul_rn(f32(qkv[t*3072+1024+channel]),c),__fmul_rn(sign*f32(qkv[t*3072+1024+partner]),s)));
}
__global__ void gelu_kernel(bf16* x,size_t n,int exact) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    float a=f32(x[i]);
    float v=exact?0.5f*a*(1.f+erff(a*0.7071067811865475244f)):
        0.5f*a*(1.f+tanhf(0.7978845608028654f*(a+0.044715f*a*a*a)));
    x[i]=to_bf16(v);
}
__global__ void add_kernel(bf16* x,const bf16* delta,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=to_bf16(f32(x[i])+f32(delta[i]));
}
__global__ void bias_kernel(bf16* x,const bf16* bias,size_t n,int d) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=to_bf16(f32(x[i])+f32(bias[i%d]));
}
__global__ void to_float_kernel(const bf16* x,float* out,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=f32(x[i]);
}
__global__ void lora_add_kernel(bf16* x,const float* delta,size_t n,float scale) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=to_bf16(__fadd_rn(f32(x[i]),__fmul_rn(scale,delta[i])));
}
} }
using namespace cs1;
extern "C" int cs1_vision_norm(const void* x,const void* w,const void* b,void* y,int rows,int d,void* stream) {
    if(rows<=0 || d<=0) return cudaErrorInvalidValue;
    vision::norm_kernel<<<rows,256,0,(cudaStream_t)stream>>>((const bf16*)x,(const bf16*)w,(const bf16*)b,(bf16*)y,d);
    return cudaGetLastError();
}
extern "C" int cs1_vision_position(void* x,const void* table,const int* indices,const float* weights,int n,void* stream) {
    if(n<=0) return cudaErrorInvalidValue;
    vision::position_kernel<<<(n*1024+255)/256,256,0,(cudaStream_t)stream>>>((bf16*)x,(const bf16*)table,indices,weights,(size_t)n*1024);
    return cudaGetLastError();
}
extern "C" int cs1_vision_rope(const void* qkv,const float* co,const float* si,void* q,void* k,int n,void* stream) {
    if(n<=0) return cudaErrorInvalidValue;
    vision::rope_kernel<<<(n*1024+255)/256,256,0,(cudaStream_t)stream>>>((const bf16*)qkv,co,si,(bf16*)q,(bf16*)k,(size_t)n*1024);
    return cudaGetLastError();
}
extern "C" int cs1_vision_attention(const void* q,const void* k,const void* v,void* out,int n,void* stream) {
    if(n<=0) return cudaErrorInvalidValue;
    namespace f=vision::flash;
    f::flash_kernel<64><<<dim3((n+f::BM-1)/f::BM,16),f::THREADS,f::SMEM_BYTES<64>,(cudaStream_t)stream>>>(
        (const bf16*)q,(const bf16*)k,(const bf16*)v,3072,(bf16*)out,n,16,16,0.125f*1.4426950408889634f);
    return cudaGetLastError();
}
extern "C" int cs1_vision_gelu(void* x,size_t n,int exact,void* stream) {
    if(n==0) return cudaSuccess;
    vision::gelu_kernel<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((bf16*)x,n,exact); return cudaGetLastError();
}
extern "C" int cs1_vision_add(void* x,const void* delta,size_t n,void* stream) {
    if(n==0) return cudaSuccess;
    vision::add_kernel<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((bf16*)x,(const bf16*)delta,n); return cudaGetLastError();
}
extern "C" int cs1_vision_to_float(const void* x,float* out,size_t n,void* stream) {
    if(n==0) return cudaSuccess;
    vision::to_float_kernel<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((const bf16*)x,out,n); return cudaGetLastError();
}
extern "C" int cs1_vision_lora_add(void* x,const float* delta,size_t n,float scale,void* stream) {
    if(n==0) return cudaSuccess;
    vision::lora_add_kernel<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((bf16*)x,delta,n,scale); return cudaGetLastError();
}

// cuDNN Conv3d rounds its convolution output before its separate bias addition.
extern "C" int cs1_vision_bias(void* x,const void* bias,size_t n,int d,void* stream) {
    if(d<=0) return cudaErrorInvalidValue;
    if(n==0) return cudaSuccess;
    vision::bias_kernel<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((bf16*)x,(const bf16*)bias,n,d); return cudaGetLastError();
}

namespace cs1 { namespace vision {
__global__ void position_v2_kernel(bf16* x,const bf16* table,const int* indices,const float* weights,size_t n,int hidden) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int token=i/hidden,d=i%hidden;
    float pos=0.f;
    for(int j=0;j<4;j++) pos=__fadd_rn(pos,__fmul_rn(f32(table[(size_t)indices[token*4+j]*hidden+d]),weights[token*4+j]));
    x[i]=to_bf16(f32(x[i])+round_bf16(pos));
}
__global__ void rope_v2_kernel(const bf16* qkv,const float* co,const float* si,bf16* q,bf16* k,size_t n,int hidden,int dh) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int token=i/hidden,d=i%dh,channel=i%hidden,half=dh/2;
    const float c=co[(size_t)token*half+d%half],s=si[(size_t)token*half+d%half];
    const int partner=channel+(d<half?half:-half);
    const float sign=d<half?-1.f:1.f;
    q[i]=to_bf16(__fadd_rn(__fmul_rn(f32(qkv[(size_t)token*hidden*3+channel]),c),__fmul_rn(sign*f32(qkv[(size_t)token*hidden*3+partner]),s)));
    k[i]=to_bf16(__fadd_rn(__fmul_rn(f32(qkv[(size_t)token*hidden*3+hidden+channel]),c),__fmul_rn(sign*f32(qkv[(size_t)token*hidden*3+hidden+partner]),s)));
}
// Q/K are compact; V points into interleaved QKV. Every padded column is explicitly zero.
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
extern "C" int cs1_vision_position_v2(void* x,const void* table,const int* indices,const float* weights,int n,int hidden,void* stream) {
    if(n<=0 || (hidden!=1024 && hidden!=1152)) return cudaErrorInvalidValue;
    const size_t count=(size_t)n*hidden;
    vision::position_v2_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((bf16*)x,(const bf16*)table,indices,weights,count,hidden);
    return cudaGetLastError();
}
extern "C" int cs1_vision_rope_v2(const void* qkv,const float* co,const float* si,void* q,void* k,int n,int hidden,int dh,void* stream) {
    if(n<=0 || !((hidden==1024 && dh==64)||(hidden==1152 && dh==72))) return cudaErrorInvalidValue;
    const size_t count=(size_t)n*hidden;
    vision::rope_v2_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((const bf16*)qkv,co,si,(bf16*)q,(bf16*)k,count,hidden,dh);
    return cudaGetLastError();
}
extern "C" int cs1_vision_attention_v2(const void* q,const void* k,const void* v,void* out,int n,int heads,int dh,void* workspace,void* stream) {
    if(n<=0 || heads!=16 || (dh!=64 && dh!=72)) return cudaErrorInvalidValue;
    if(dh==64) return cs1_vision_attention(q,k,v,out,n,stream);
    if(!workspace) return cudaErrorInvalidValue;
    const size_t count=(size_t)n*heads*80;
    bf16* pq=(bf16*)workspace; bf16* pk=pq+count; bf16* pv=pk+count; bf16* po=pv+count;
    vision::pad_attention_kernel<<<(count+255)/256,256,0,(cudaStream_t)stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)v,pq,pk,pv,count,heads,dh);
    cudaError_t status=cudaGetLastError(); if(status!=cudaSuccess) return status;
    namespace f=vision::flash;
    f::flash_kernel<80><<<dim3((n+f::BM-1)/f::BM,heads),f::THREADS,f::SMEM_BYTES<80>,(cudaStream_t)stream>>>(
        pq,pk,pv,heads*80,po,n,heads,heads,rsqrtf((float)dh)*1.4426950408889634f);
    status=cudaGetLastError(); if(status!=cudaSuccess) return status;
    vision::unpack_attention_kernel<<<((size_t)n*heads*dh+255)/256,256,0,(cudaStream_t)stream>>>(po,(bf16*)out,(size_t)n*heads*dh,dh);
    return cudaGetLastError();
}
