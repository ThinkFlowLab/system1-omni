#include "common.cuh"
namespace cs1 { namespace reference_prep {
constexpr int DH = 256;         // head dim
constexpr int PER = DH / 32;    // values per lane

// One warp per (token, head), q heads first, then k heads. Each lane holds 8
// consecutive dims, so the rotary partner of dim i < 32 (dim i + 32) sits in lane ^ 4.
__global__ void attn_prep_kernel(const bf16* __restrict__ qg, const bf16* __restrict__ kr, int ld,
                                 const bf16* __restrict__ qw, const bf16* __restrict__ kw,
                                 const bf16* __restrict__ cos, const bf16* __restrict__ sin,
                                 bf16* __restrict__ q, bf16* __restrict__ gate, bf16* __restrict__ k, int T, int Hq,
                                 int Hk, int half, float eps) {
    const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    const int heads = Hq + Hk;
    if (warp >= T * heads) return;
    const int t = warp / heads, hh = warp % heads;
    const bool is_q = hh < Hq;
    const int h = is_q ? hh : hh - Hq;
    const bf16* src = is_q ? qg + (size_t)t * ld + (size_t)h * 2 * DH : kr + (size_t)t * ld + (size_t)h * DH;
    const bf16* w = is_q ? qw : kw;
    const int d0 = lane * PER;

    float x[PER];
    load8(src + d0, x);
    float sums[4]={};
#pragma unroll
    for(int j=0;j<4;j++) {
        float a=f32(src[lane*4+j]),b=f32(src[128+lane*4+j]);
        sums[j]=__fadd_rn(__fmul_rn(a,a),__fmul_rn(b,b));
    }
    float ss=__fadd_rn(__fadd_rn(__fadd_rn(sums[0],sums[1]),sums[2]),sums[3]);
    for(int step=16;step>0;step/=2)ss=__fadd_rn(ss,__shfl_down_sync(0xffffffff,ss,step));
    ss=__shfl_sync(0xffffffff,ss,0);
    const float inv=rsqrtf(__fadd_rn(__fmul_rn(ss,1.f/DH),eps));
    float wv[PER];
    load8(w + d0, wv);
#pragma unroll
    for (int i = 0; i < PER; i++) x[i] = round_bf16(x[i] * inv * (1.f + wv[i]));

    // rotate_half on the first 2 * half dims: out = x * cos + rotate_half(x) * sin,
    // each product and the sum rounded to bfloat16 as in the reference.
    const int rot = 2 * half;
    float y[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const float partner = __shfl_xor_sync(0xffffffffu, x[i], (half / PER));
        const int d = d0 + i;
        y[i] = x[i];
        if (d < rot) {
            const int fi = d % half;
            const float c = f32(cos[(size_t)t * half + fi]), s = f32(sin[(size_t)t * half + fi]);
            const float r = d < half ? -partner : partner;
            y[i] = round_bf16(round_bf16(x[i] * c) + round_bf16(r * s));
        }
    }
    if (is_q) {
        store8(q + ((size_t)t * Hq + h) * DH + d0, y);
        float gv[PER];
        load8(src + DH + d0, gv);
        store8(gate + ((size_t)t * Hq + h) * DH + d0, gv);
    } else {
        store8(k + ((size_t)t * Hk + h) * DH + d0, y);
    }
}

} }
using namespace cs1;
using namespace cs1::reference_prep;
extern "C" int cs1_reference_attn_prep(const void* qg, const void* kr, int ld, const void* qw, const void* kw,
                             const void* cos, const void* sin, void* q, void* gate, void* k, int T, int Hq, int Hk,
                             int Dh, int half, float eps, void* stream) {
    if (ld % 8 != 0 || T < 0) return cudaErrorInvalidValue;
    // the lane ^ (half / 8) partner exchange needs 2 * half <= 256 and half a multiple of 8
    if (Dh != DH || half % PER != 0 || 2 * half > DH || (half / PER) & ((half / PER) - 1)) return cudaErrorInvalidValue;
    const int warps = T * (Hq + Hk);
    if (warps == 0) return cudaSuccess;
    constexpr int WARPS = 8;
    attn_prep_kernel<<<(warps + WARPS - 1) / WARPS, WARPS * 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(qg), static_cast<const bf16*>(kr), ld, static_cast<const bf16*>(qw),
        static_cast<const bf16*>(kw), static_cast<const bf16*>(cos), static_cast<const bf16*>(sin),
        static_cast<bf16*>(q), static_cast<bf16*>(gate), static_cast<bf16*>(k), T, Hq, Hk, half, eps);
    return cudaGetLastError();
}
