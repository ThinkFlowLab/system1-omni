// Chunked Gated DeltaNet prefill (forward only, batch 1) on tensor cores, sm_80 and later.
//
// The math of torch_chunk_gated_delta_rule in Transformers (chunks of 64), split into
// three kernels the way flash-linear-attention splits its chunked forward pass:
//   1. gdn_chunk_prep, per (chunk, head): L2 norms, the cumulative decays, the pair
//      products k.k and q.k (TF32 with float32 accumulation), the triangular inverse
//      T = (I + A)^-1 (float32, CUDA cores), and u = T (beta v), w = T (beta exp(cum) k)
//      (bfloat16 with float32 accumulation). Results are stored as bfloat16.
//   2. gdn_chunk_state, per (head, 32 value columns), over the chunks in order: keeps
//      the state S in float32 registers, stores it as bfloat16 before each chunk, and
//      computes v_new = u - w S and S = decay S + kd^T v_new with mma.sync.
//   3. gdn_chunk_out, per (chunk, head): o = qd S + P v_new with mma.sync.
// Transformers computes all of this in float32. Keeping the intermediate results in
// bfloat16, as flash-linear-attention does, makes this kernel less precise than that
// path; tests/kernels.rs checks it against a float64 token-by-token reference.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include "mma.cuh"
#include "ops.h"

using namespace nvcuda;

namespace {

using cs1::bf16;
using cs1::warp_sum;

constexpr int C = 64;    // chunk length
constexpr int K = 128;   // key head dim
constexpr int V = 128;   // value head dim
constexpr int THREADS = 256;
// row strides: multiples of 16 bytes as WMMA needs, and not multiples of 32 floats
constexpr int KP = K + 4;   // float
constexpr int CP = C + 4;   // float
constexpr int HB = K + 8;   // bfloat16
constexpr int TB = C + 8;   // bfloat16

using ATf32Row = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using BTf32Col = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
using CTf32 = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;
using ABf16 = wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>;
using BBf16 = wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major>;
using CBf16 = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <typename F>
__device__ __forceinline__ void to_tf32(F& f) {
#pragma unroll
    for (int t = 0; t < f.num_elements; t++) f.x[t] = wmma::__float_to_tf32(f.x[t]);
}

struct Work {
    bf16* u;      // [H, NCC, V]
    bf16* w;      // [H, NCC, K]
    bf16* qd;     // [H, NCC, K]  q * scale * exp(cum)
    bf16* kd;     // [H, NCC, K]  k * exp(cum_last - cum)
    float* p;     // [H, NC, C, C] float32 scratch of q.k
    bf16* pb;     // [H, NC, C, C] (q.k) exp(cum_i - cum_j), j <= i
    float* decay; // [H, NC]       exp(cum_last)
    bf16* s;      // [H, NC, K, V] state before each chunk
    bf16* vn;     // [H, NCC, V]   v_new
};

// Byte offsets of the workspace parts, 256-byte aligned.
struct Layout {
    size_t u, w, qd, kd, p, pb, decay, s, vn, total;
    Layout(int T, int H) {
        const size_t NC = (T + C - 1) / C, NCC = NC * C;
        size_t at = 0;
        auto take = [&](size_t bytes) {
            const size_t off = at;
            at = (at + bytes + 255) / 256 * 256;
            return off;
        };
        u = take((size_t)H * NCC * V * 2);
        w = take((size_t)H * NCC * K * 2);
        qd = take((size_t)H * NCC * K * 2);
        kd = take((size_t)H * NCC * K * 2);
        p = take((size_t)H * NC * C * C * 4);
        pb = take((size_t)H * NC * C * C * 2);
        decay = take((size_t)H * NC * 4);
        s = take((size_t)H * NC * K * V * 2);
        vn = take((size_t)H * NCC * V * 2);
        total = at;
    }
};

// kernel 1 shared memory, bytes
constexpr int R1 = 0;                     // kn float [C][KP]; later v as bf16 [C][HB]
constexpr int R2 = R1 + C * KP * 4;       // qn float [C][KP]; later T float [C][CP] + scratch, then T as bf16
constexpr int R3 = R2 + C * KP * 4;       // A float [C][CP]; later k exp(cum) as bf16 [C][HB]
constexpr int R4 = R3 + C * CP * 4;       // cum, beta, exp(cum), exp(cum_last - cum)
constexpr int R5 = R4 + 4 * C * 4;        // per-warp 16x16 float staging of u and w
constexpr size_t SMEM1_BYTES = R5 + (THREADS / 32) * 256 * 4;
constexpr int TBF = C * CP * 4 + 3 * 256 * 4;  // offset of T as bf16 inside R2
static_assert(C * HB * 2 <= C * CP * 4, "k exp(cum) as bf16 fits over A");
static_assert(TBF + C * TB * 2 <= C * KP * 4, "T as bf16 fits in R2");

__global__ void __launch_bounds__(THREADS) gdn_chunk_prep(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, Work ws, int T, int H, int HK, float scale) {
    extern __shared__ __align__(128) unsigned char sm[];
    float* kn = reinterpret_cast<float*>(sm + R1);
    float* qn = reinterpret_cast<float*>(sm + R2);
    float* tm = qn;
    float* sc = tm + C * CP;
    float* am = reinterpret_cast<float*>(sm + R3);
    float* cum = reinterpret_cast<float*>(sm + R4);
    float* bet = cum + C;
    float* ecum = bet + C;
    float* erem = ecum + C;
    __nv_bfloat16* vb = reinterpret_cast<__nv_bfloat16*>(sm + R1);
    __nv_bfloat16* tb = reinterpret_cast<__nv_bfloat16*>(sm + R2 + TBF);
    __nv_bfloat16* kw = reinterpret_cast<__nv_bfloat16*>(sm + R3);

    const int c = blockIdx.x, h = blockIdx.y, NC = gridDim.x, NCC = NC * C;
    const int hk = h / (H / HK);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // 1. load q and k rows of this chunk, L2-normalize (padding rows stay zero)
    for (int r = warp; r < C; r += THREADS / 32) {
        const int t = c * C + r;
        float kv[4], qv[4], ks = 0.f, qs = 0.f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int d = lane + 32 * e;
            float kx = 0.f, qx = 0.f;
            if (t < T) {
                kx = __bfloat162float(k[((size_t)t * HK + hk) * K + d]);
                qx = __bfloat162float(q[((size_t)t * HK + hk) * K + d]);
            }
            kv[e] = kx;
            qv[e] = qx;
            ks += kx * kx;
            qs += qx * qx;
        }
        ks = warp_sum(ks);
        qs = warp_sum(qs);
        const float kinv = rsqrtf(ks + 1e-6f), qinv = rsqrtf(qs + 1e-6f) * scale;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int d = lane + 32 * e;
            kn[r * KP + d] = kv[e] * kinv;
            qn[r * KP + d] = qv[e] * qinv;
        }
    }
    if (tid < C) {
        const int t = c * C + tid;
        cum[tid] = t < T ? g[(size_t)t * H + h] : 0.f;
        bet[tid] = t < T ? __bfloat162float(beta[(size_t)t * H + h]) : 0.f;
    }
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int i = 0; i < C; i++) {
            s += cum[i];
            cum[i] = s;
        }
        ws.decay[h * NC + c] = expf(s);
    }
    __syncthreads();
    if (tid < C) {
        ecum[tid] = expf(cum[tid]);
        erem[tid] = expf(cum[C - 1] - cum[tid]);
    }
    __syncthreads();

    // 2. decayed q and k for the state and output kernels
    for (int x = tid; x < C * K; x += THREADS) {
        const int i = x / K, d = x % K;
        const size_t row = ((size_t)h * NCC + c * C + i) * K + d;
        ws.qd[row] = __float2bfloat16(qn[i * KP + d] * ecum[i]);
        ws.kd[row] = __float2bfloat16(kn[i * KP + d] * erem[i]);
    }

    // 3. pair products on tensor cores: the 10 lower 16x16 tiles of k.k (into A) and of
    //    q.k (into P, in global memory), then the masks and decays elementwise
    float* pout = ws.p + ((size_t)h * NC + c) * C * C;
    for (int e = warp; e < 20; e += THREADS / 32) {
        const int tile = e % 10;
        const int it = tile < 1 ? 0 : tile < 3 ? 1 : tile < 6 ? 2 : 3;
        const int jt = tile - it * (it + 1) / 2;
        const float* a = (e < 10 ? kn : qn) + it * 16 * KP;
        const float* b = kn + jt * 16 * KP;
        CTf32 acc;
        wmma::fill_fragment(acc, 0.f);
#pragma unroll 4
        for (int k0 = 0; k0 < K; k0 += 8) {
            ATf32Row fa;
            BTf32Col fb;
            wmma::load_matrix_sync(fa, a + k0, KP);
            wmma::load_matrix_sync(fb, b + k0, KP);
            to_tf32(fa);
            to_tf32(fb);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        if (e < 10)
            wmma::store_matrix_sync(am + it * 16 * CP + jt * 16, acc, CP, wmma::mem_row_major);
        else
            wmma::store_matrix_sync(pout + it * 16 * C + jt * 16, acc, C, wmma::mem_row_major);
    }
    __syncthreads();
    __nv_bfloat16* pb = ws.pb + ((size_t)h * NC + c) * C * C;
    for (int x = tid; x < C * C; x += THREADS) {
        const int i = x / C, j = x % C;
        const float dec = j <= i ? expf(cum[i] - cum[j]) : 0.f;
        am[i * CP + j] = j < i ? bet[i] * am[i * CP + j] * dec : 0.f;
        pb[i * C + j] = __float2bfloat16(j <= i ? pout[i * C + j] * dec : 0.f);
    }
    __syncthreads();

    // 4. T = (I + A)^-1 in 16x16 blocks
    for (int x = tid; x < C * CP; x += THREADS) tm[x] = 0.f;
    __syncthreads();
    if (tid < C) {
        const int base = 16 * (tid / 16), col = tid % 16;
        float xs[16];
#pragma unroll
        for (int r = 0; r < 16; r++) {
            float acc = r == col ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < r; j++) acc = fmaf(-am[(base + r) * CP + base + j], xs[j], acc);
            xs[r] = acc;
        }
#pragma unroll
        for (int r = 0; r < 16; r++) tm[(base + r) * CP + base + col] = xs[r];
    }
    __syncthreads();
    for (int lv = 1; lv < 4; lv++) {
        const int nb = 4 - lv;
        for (int x = tid; x < nb * 256; x += THREADS) {
            const int bi = lv + x / 256, bj = bi - lv, r = (x % 256) / 16, cc = x % 16;
            float acc = 0.f;
            for (int kb = bj; kb < bi; kb++)
#pragma unroll
                for (int m = 0; m < 16; m++)
                    acc = fmaf(am[(16 * bi + r) * CP + 16 * kb + m], tm[(16 * kb + m) * CP + 16 * bj + cc], acc);
            sc[x] = acc;
        }
        __syncthreads();
        for (int x = tid; x < nb * 256; x += THREADS) {
            const int bi = lv + x / 256, bj = bi - lv, r = (x % 256) / 16, cc = x % 16;
            const float* s0 = sc + (x / 256) * 256 + cc;
            float acc = 0.f;
            for (int m = 0; m <= r; m++) acc = fmaf(tm[(16 * bi + r) * CP + 16 * bi + m], s0[m * 16], acc);
            tm[(16 * bi + r) * CP + 16 * bj + cc] = -acc;
        }
        __syncthreads();
    }

    // 5. bfloat16 operands: k exp(cum) over A, T beta after T, then v over kn;
    //    u = T (beta v) and w = T (beta exp(cum) k) on tensor cores, stored as bfloat16
    for (int x = tid; x < C * K; x += THREADS) {
        const int j = x / K, d = x % K;
        kw[j * HB + d] = __float2bfloat16(kn[j * KP + d] * ecum[j]);
    }
    for (int x = tid; x < C * C; x += THREADS) {
        const int i = x / C, j = x % C;
        tb[i * TB + j] = __float2bfloat16(tm[i * CP + j] * bet[j]);
    }
    __syncthreads();
    for (int x = tid; x < C * V; x += THREADS) {
        const int j = x / V, d = x % V;
        const int t = c * C + j;
        vb[j * HB + d] = t < T ? v[((size_t)t * H + h) * V + d] : __float2bfloat16(0.f);
    }
    __syncthreads();
    float* stage = reinterpret_cast<float*>(sm + R5) + warp * 256;
    for (int e = warp; e < 64; e += THREADS / 32) {
        const bool is_u = e < 32;
        const int it = (e % 32) / 8, dt = e % 8;
        const __nv_bfloat16* bsrc = is_u ? vb : kw;
        CBf16 acc;
        wmma::fill_fragment(acc, 0.f);
        for (int kb = 0; kb <= it; kb++) {  // T is lower triangular
            ABf16 fa;
            BBf16 fb;
            wmma::load_matrix_sync(fa, tb + it * 16 * TB + kb * 16, TB);
            wmma::load_matrix_sync(fb, bsrc + kb * 16 * HB + dt * 16, HB);
            wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
        __syncwarp();
        bf16* dst = (is_u ? ws.u : ws.w) + ((size_t)h * NCC + c * C + it * 16) * K + dt * 16;
        for (int x = lane; x < 256; x += 32) dst[(x / 16) * K + x % 16] = __float2bfloat16(stage[x]);
        __syncwarp();
    }
}

// ---- kernel 2: the state, chunk by chunk ----

constexpr int BVS = 32;             // value columns per block
constexpr int ST_THREADS = 128;
constexpr int WS_LD = K + 8;        // bfloat16 row stride of staged w and kd
constexpr int SS_LD = BVS + 8;      // bfloat16 row stride of the S copy and v_new
constexpr int STAGE = C * WS_LD;    // elements of one staged w or kd
constexpr size_t SMEM2_BYTES = (4 * STAGE + K * SS_LD + C * SS_LD) * 2;

__global__ void __launch_bounds__(ST_THREADS) gdn_chunk_state(Work ws, int NC) {
    extern __shared__ __align__(128) unsigned char sm[];
    bf16* wbuf = reinterpret_cast<bf16*>(sm);  // [2][C][WS_LD]
    bf16* kbuf = wbuf + 2 * STAGE;              // [2][C][WS_LD]
    bf16* scopy = kbuf + 2 * STAGE;             // [K][SS_LD]
    bf16* vnew = scopy + K * SS_LD;             // [C][SS_LD]
    const int h = blockIdx.x, vb0 = blockIdx.y * BVS, NCC = NC * C;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;

    auto load = [&](int c, int buf) {
        const size_t base = ((size_t)h * NCC + (size_t)c * C) * K;
        for (int x = tid; x < C * K / 8; x += ST_THREADS) {
            const int r = x / (K / 8), col = (x % (K / 8)) * 8;
            cs1::cp_async16(wbuf + buf * STAGE + r * WS_LD + col, ws.w + base + (size_t)r * K + col);
            cs1::cp_async16(kbuf + buf * STAGE + r * WS_LD + col, ws.kd + base + (size_t)r * K + col);
        }
        cs1::cp_async_commit();
    };

    // S rows warp * 32 + mt * 16 + {g, g + 8}, columns nt * 8 + {2t, 2t + 1}
    float st[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; mt++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++) st[mt][nt][0] = st[mt][nt][1] = st[mt][nt][2] = st[mt][nt][3] = 0.f;

    load(0, 0);
    for (int c = 0; c < NC; c++) {
        const int buf = c & 1;
        cs1::cp_async_wait<0>();
        __syncthreads();  // chunk c is staged, and chunk c - 1 is done with the other buffer
        if (c + 1 < NC) load(c + 1, buf ^ 1);
        // 1. S as bfloat16, to shared memory for w S and to global memory for the output
        bf16* sg = ws.s + ((size_t)h * NC + c) * K * V + vb0;
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int row = warp * 32 + mt * 16 + g + r * 8, col = nt * 8 + 2 * t;
                    const uint32_t pk = cs1::pack_bf16(st[mt][nt][2 * r], st[mt][nt][2 * r + 1]);
                    *reinterpret_cast<uint32_t*>(scopy + row * SS_LD + col) = pk;
                    *reinterpret_cast<uint32_t*>(sg + (size_t)row * V + col) = pk;
                }
        __syncthreads();
        // 2. v_new = u - w S, rows warp * 16 ..
        const bf16* wc = wbuf + buf * STAGE;
        float acc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < K; kk += 16) {
            uint32_t a[4];
            cs1::load_a(a, wc, WS_LD, warp * 16, kk, lane);
#pragma unroll
            for (int nt = 0; nt < 4; nt += 2) {
                uint32_t b[4];
                cs1::load_b_kn(b, scopy, SS_LD, kk, nt * 8, lane);
                cs1::mma16816(acc[nt], a, b[0], b[1]);
                cs1::mma16816(acc[nt + 1], a, b[2], b[3]);
            }
        }
        const size_t row0 = (size_t)h * NCC + (size_t)c * C + warp * 16;
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int r = 0; r < 2; r++) {
                const int row = g + r * 8, col = nt * 8 + 2 * t;
                const __nv_bfloat162 uv =
                    *reinterpret_cast<const __nv_bfloat162*>(ws.u + (row0 + row) * V + vb0 + col);
                const uint32_t pk = cs1::pack_bf16(__low2float(uv) - acc[nt][2 * r], __high2float(uv) - acc[nt][2 * r + 1]);
                *reinterpret_cast<uint32_t*>(vnew + (warp * 16 + row) * SS_LD + col) = pk;
                *reinterpret_cast<uint32_t*>(ws.vn + (row0 + row) * V + vb0 + col) = pk;
            }
        __syncthreads();
        // 3. S = decay S + kd^T v_new, rows warp * 32 ..
        const float dec = ws.decay[h * NC + c];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) st[mt][nt][e] *= dec;
        const bf16* kc = kbuf + buf * STAGE;
#pragma unroll
        for (int kk = 0; kk < C; kk += 16) {
#pragma unroll
            for (int mt = 0; mt < 2; mt++) {
                uint32_t a[4];
                cs1::load_a_trans(a, kc, WS_LD, warp * 32 + mt * 16, kk, lane);
#pragma unroll
                for (int nt = 0; nt < 4; nt += 2) {
                    uint32_t b[4];
                    cs1::load_b_kn(b, vnew, SS_LD, kk, nt * 8, lane);
                    cs1::mma16816(st[mt][nt], a, b[0], b[1]);
                    cs1::mma16816(st[mt][nt + 1], a, b[2], b[3]);
                }
            }
        }
    }
}

// ---- kernel 3: the output, per chunk ----

constexpr int OUT_THREADS = 128;
constexpr int O_LD = K + 8;  // bfloat16 row stride of qd, S and v_new (all 128 wide)
constexpr int P_LD = C + 8;  // bfloat16 row stride of P
constexpr size_t SMEM3_BYTES = (C * O_LD + K * O_LD + C * P_LD + C * O_LD) * 2;
static_assert(K == V, "S, qd and v_new share a row stride");

__global__ void __launch_bounds__(OUT_THREADS) gdn_chunk_out(Work ws, bf16* __restrict__ o, int T, int H) {
    extern __shared__ __align__(128) unsigned char sm[];
    bf16* qs = reinterpret_cast<bf16*>(sm);  // [C][O_LD]
    bf16* ss = qs + C * O_LD;                // [K][O_LD]
    bf16* ps = ss + K * O_LD;                // [C][P_LD]
    bf16* vs = ps + C * P_LD;                // [C][O_LD]
    const int c = blockIdx.x, h = blockIdx.y, NC = gridDim.x, NCC = NC * C;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
    const size_t rows = (size_t)h * NCC + (size_t)c * C;
    for (int x = tid; x < C * K / 8; x += OUT_THREADS) {
        const int r = x / (K / 8), col = (x % (K / 8)) * 8;
        cs1::cp_async16(qs + r * O_LD + col, ws.qd + (rows + r) * K + col);
        cs1::cp_async16(vs + r * O_LD + col, ws.vn + (rows + r) * V + col);
    }
    for (int x = tid; x < K * V / 8; x += OUT_THREADS) {
        const int r = x / (V / 8), col = (x % (V / 8)) * 8;
        cs1::cp_async16(ss + r * O_LD + col, ws.s + ((size_t)h * NC + c) * K * V + (size_t)r * V + col);
    }
    for (int x = tid; x < C * C / 8; x += OUT_THREADS) {
        const int r = x / (C / 8), col = (x % (C / 8)) * 8;
        cs1::cp_async16(ps + r * P_LD + col, ws.pb + ((size_t)h * NC + c) * C * C + r * C + col);
    }
    cs1::cp_async_commit();
    cs1::cp_async_wait<0>();
    __syncthreads();

    float acc[V / 8][4];
#pragma unroll
    for (int nt = 0; nt < V / 8; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    // qd S
#pragma unroll 2
    for (int kk = 0; kk < K; kk += 16) {
        uint32_t a[4];
        cs1::load_a(a, qs, O_LD, warp * 16, kk, lane);
#pragma unroll
        for (int nt = 0; nt < V / 8; nt += 2) {
            uint32_t b[4];
            cs1::load_b_kn(b, ss, O_LD, kk, nt * 8, lane);
            cs1::mma16816(acc[nt], a, b[0], b[1]);
            cs1::mma16816(acc[nt + 1], a, b[2], b[3]);
        }
    }
    // P v_new; P is lower triangular, so these rows need positions below (warp + 1) * 16
    for (int kk = 0; kk < (warp + 1) * 16; kk += 16) {
        uint32_t a[4];
        cs1::load_a(a, ps, P_LD, warp * 16, kk, lane);
#pragma unroll
        for (int nt = 0; nt < V / 8; nt += 2) {
            uint32_t b[4];
            cs1::load_b_kn(b, vs, O_LD, kk, nt * 8, lane);
            cs1::mma16816(acc[nt], a, b[0], b[1]);
            cs1::mma16816(acc[nt + 1], a, b[2], b[3]);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int tok = c * C + warp * 16 + g + r * 8;
        if (tok >= T) continue;
        bf16* dst = o + ((size_t)tok * H + h) * V + 2 * t;
#pragma unroll
        for (int nt = 0; nt < V / 8; nt++)
            *reinterpret_cast<uint32_t*>(dst + nt * 8) = cs1::pack_bf16(acc[nt][2 * r], acc[nt][2 * r + 1]);
    }
}

Work split(float* workspace, int T, int H) {
    const Layout l(T, H);
    unsigned char* b = reinterpret_cast<unsigned char*>(workspace);
    Work ws;
    ws.u = reinterpret_cast<bf16*>(b + l.u);
    ws.w = reinterpret_cast<bf16*>(b + l.w);
    ws.qd = reinterpret_cast<bf16*>(b + l.qd);
    ws.kd = reinterpret_cast<bf16*>(b + l.kd);
    ws.p = reinterpret_cast<float*>(b + l.p);
    ws.pb = reinterpret_cast<bf16*>(b + l.pb);
    ws.decay = reinterpret_cast<float*>(b + l.decay);
    ws.s = reinterpret_cast<bf16*>(b + l.s);
    ws.vn = reinterpret_cast<bf16*>(b + l.vn);
    return ws;
}

}  // namespace

extern "C" {

size_t cs1_gdn_workspace_floats(int T, int H) { return (Layout(T, H).total + 3) / 4; }

int cs1_gdn_prefill(const void* q, const void* k, const void* v, const float* g, const void* beta,
                    void* o, float* workspace, int T, int H, int HK, float scale, void* stream) {
    if (T < 0 || HK <= 0 || H % HK != 0) return cudaErrorInvalidValue;
    if (T == 0) return cudaSuccess;
    // once per process (for the device current at the first call)
    static const cudaError_t configured = [] {
        cudaError_t e = cudaFuncSetAttribute(gdn_chunk_prep, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             (int)SMEM1_BYTES);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(gdn_chunk_state, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)SMEM2_BYTES);
        if (e == cudaSuccess)
            e = cudaFuncSetAttribute(gdn_chunk_out, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)SMEM3_BYTES);
        return e;
    }();
    if (configured != cudaSuccess) return configured;
    const int NC = (T + C - 1) / C;
    const Work ws = split(workspace, T, H);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    gdn_chunk_prep<<<dim3(NC, H), THREADS, SMEM1_BYTES, st>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v), g, static_cast<const __nv_bfloat16*>(beta), ws, T, H, HK, scale);
    gdn_chunk_state<<<dim3(H, V / BVS), ST_THREADS, SMEM2_BYTES, st>>>(ws, NC);
    gdn_chunk_out<<<dim3(NC, H), OUT_THREADS, SMEM3_BYTES, st>>>(ws, static_cast<bf16*>(o), T, H);
    return cudaGetLastError();
}

}  // extern "C"
