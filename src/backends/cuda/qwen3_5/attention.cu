// Full-attention layers of Qwen3.5: q/k preparation and causal attention.
//
// cs1_attention is a FlashAttention-2 style kernel on tensor cores (mma.sync
// m16n8k16, bfloat16 in, float32 accumulation): a block takes 64 queries of one head,
// four warps of 16 rows each, and walks the keys up to its last query in tiles of
// 32, keeping the output and the online softmax in registers. The probabilities are
// rounded to bfloat16 for the P*V product, as in flash attention; the running sums
// stay float32.
#include "common.cuh"
#include "mma.cuh"
#include "ops.h"

namespace cs1 {
namespace {

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
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < PER; i++) ss += x[i] * x[i];
    const float inv = rsqrtf(warp_sum(ss) / DH + eps);
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

// ---- flash attention ----

namespace flash {

constexpr int D = 256, BM = 64, BN = 32, THREADS = 128;
constexpr int LDS = D + 8;  // shared row stride in elements: 528 bytes keeps ldmatrix conflict-free
constexpr int SMEM_BYTES = (BM + 2 * BN) * LDS * 2;

__global__ void __launch_bounds__(THREADS)
    flash_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v, int ldv,
                 bf16* __restrict__ out, int T, int Hq, int Hk, float scale_log2) {
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

    const int kv_end = min(T, q0 + BM);
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
        const bool active = k0 <= row0 + 15;
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
                    const int key = k0 + n * 8 + 2 * t + (e & 1), row = row0 + g + (e >> 1) * 8;
                    sc[n][e] = (key <= row && key < T) ? sc[n][e] * scale_log2 : -INFINITY;
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

}  // namespace
}  // namespace cs1

using namespace cs1;

extern "C" int cs1_attn_prep(const void* qg, const void* kr, int ld, const void* qw, const void* kw,
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

extern "C" int cs1_attention(const void* q, const void* k, const void* v, int ldv, void* out, int T, int Hq, int Hk,
                             int Dh, float scale, void* stream) {
    if (Dh != flash::D || Hk <= 0 || Hq % Hk != 0 || ldv % 8 != 0 || ldv < Hk * Dh || T < 0)
        return cudaErrorInvalidValue;
    if (T == 0) return cudaSuccess;
    // once per process (for the device current at the first call)
    static const cudaError_t configured = cudaFuncSetAttribute(
        flash::flash_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, flash::SMEM_BYTES);
    if (configured != cudaSuccess) return configured;
    constexpr float LOG2E = 1.4426950408889634f;
    flash::flash_kernel<<<dim3((T + flash::BM - 1) / flash::BM, Hq), flash::THREADS, flash::SMEM_BYTES,
                          static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(q), static_cast<const bf16*>(k), static_cast<const bf16*>(v), ldv,
        static_cast<bf16*>(out), T, Hq, Hk, scale * LOG2E);
    return cudaGetLastError();
}
