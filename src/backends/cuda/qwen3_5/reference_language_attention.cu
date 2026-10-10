#include "reference_attention.cuh"
#include "common.cuh"
#include "mma.cuh"
namespace cs1 { namespace language_reference {
constexpr int BM = 64, THREADS = 128;
template<int D,int BN> constexpr int SMEM_BYTES = (BM + 2 * BN) * (D + 8) * 2;

template<int D,int BN>
__global__ void __launch_bounds__(THREADS)
    flash_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, int ldv,
                 float* __restrict__ out, float* __restrict__ lse, int T, int Hq, int Hk, float scale, int splits) {
    constexpr int LDS = D + 8;
    const float scale_log2=(float)((double)scale*1.4426950408889634);
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

    int per_split=((T+BN-1)/BN+splits-1)/splits;
    int first=blockIdx.z*per_split*BN;
    int last=min(min((T+BN-1)/BN*BN,q0+BM),first+per_split*BN);
    for (int k0=last-BN;k0>=first;k0-=BN) {
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
        const bool active = k0 <= row0+15;
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
                    sc[n][e] = (key < T && key <= row0+g+(e>>1)*8) ? sc[n][e] : -INFINITY;
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
                alpha[r] = exp2f((m[r] - base[r])*scale_log2);
                m[r] = mn;
                l[r] *= alpha[r];
            }
#pragma unroll
            for (int n = 0; n < BN / 8; n++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    sc[n][e] = exp2f(__fmul_rn(sc[n][e],scale_log2)-__fmul_rn(base[e >> 1],scale_log2));
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
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
    }
    const float inv[2] = {l[0]==0.f ? 1.f : 1.f/l[0], l[1]==0.f ? 1.f : 1.f/l[1]};
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int row = row0 + g + r * 8;
        if (row >= T) continue;
        float* dst=out+(((size_t)blockIdx.z*T+row)*Hq+h)*D+2*t;
        if(t==0) lse[((size_t)blockIdx.z*T+row)*Hq+h]=l[r]==0.f ? -INFINITY : fmaf(m[r],scale,__logf(l[r]));
#pragma unroll
        for(int n=0;n<D/8;n++) {
            dst[n*8]=o[n][2*r]*inv[r];
            dst[n*8+1]=o[n][2*r+1]*inv[r];
        }
    }
}



__global__ void combine(const float* partial,const float* lse,bf16* out,const bf16* gate,int T,int H,int D,int splits) {
    size_t index=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)T*H*D;
    if(index>=total) return;
    if(splits==1) { float result=partial[index];if(gate) result=round_bf16(result)*round_bf16(sigmoid(f32(gate[index])));out[index]=to_bf16(result);return; }
    size_t row=index/D,rows=(size_t)T*H;
    float maximum=-INFINITY,values[128];int power=1;while(power<splits) power*=2;
    for(int i=0;i<splits;i++) maximum=fmaxf(maximum,lse[(size_t)i*rows+row]);
    for(int i=0;i<power;i++) values[i]=i<splits ? expf(lse[(size_t)i*rows+row]-maximum) : 0.f;
    for(int step=power/2;step>0;step/=2) for(int i=0;i<step;i++) values[i]+=values[i+step];
    float logsum=logf(values[0])+maximum,result=0.f;
    for(int i=0;i<splits;i++) result=fmaf(expf(lse[(size_t)i*rows+row]-logsum),partial[(size_t)i*total+index],result);
    if(gate) result=round_bf16(result)*round_bf16(sigmoid(f32(gate[index])));
    out[index]=to_bf16(result);
}
} }
extern "C" int cs1_reference_attention(const void* q,const void* k,const void* v,int ldv,const void* gate,void* out,int T,int H,int HK,float scale,void* workspace,void* stream) {
    using namespace cs1;namespace f=language_reference;
    const int sms=reference::sm_count();
    if(T<=0 || T>65536 || H<=0 || !workspace || sms<=0) return cudaErrorInvalidValue;
    int splits=reference::split_count(T,H,64,sms);
    float* memory=(float*)workspace;size_t count=(size_t)splits*T*H*256;
    cudaError_t status=cudaSuccess;
    float* lse=memory+count;
    cudaFuncSetAttribute(f::flash_kernel<256,64>,cudaFuncAttributeMaxDynamicSharedMemorySize,f::SMEM_BYTES<256,64>);
    f::flash_kernel<256,64><<<dim3((T+63)/64,H,splits),128,f::SMEM_BYTES<256,64>,(cudaStream_t)stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)v,ldv,memory,lse,T,H,HK,scale,splits);
    status=cudaGetLastError();
    if(status==cudaSuccess) {
        f::combine<<<((size_t)T*H*256+127)/128,128,0,(cudaStream_t)stream>>>(memory,lse,(bf16*)out,(const bf16*)gate,T,H,256,splits);
        status=cudaGetLastError();
    }
    return status;
}

extern "C" size_t cs1_reference_language_attention_workspace_floats(int capacity,int heads) { return cs1::reference::workspace_floats(capacity,heads,256,64); }
