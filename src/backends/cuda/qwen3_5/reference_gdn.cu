// Optional JEMM Gated DeltaNet arithmetic, matched to the pinned active FLA path.
// Chunks of 64: BF16 Q/K products, FP32 diagonal inverse, layout-specific TF32
// off-diagonal inverse products, BF16 U/W and recurrent-state snapshots.
// Rounding and reduction order are intentional: rare BF16 changes in early
// layers otherwise amplify across the 64-layer JEMM language model.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include "mma.cuh"
#include "ops.h"

using namespace nvcuda;

namespace {
__device__ __forceinline__ float reference_inverse_norm(float squared) {
    float root,inverse;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(squared));
    asm("div.full.f32 %0, %1, %2;" : "=f"(inverse) : "f"(1.f), "f"(root));
    return inverse;
}
// Triton operand layouts select even/odd K order for the first product and
// natural K order for the second. Keep multiple products in one accumulator.
template<bool EvenOdd>
__device__ __forceinline__ void reference_dot16(const float* a,int lda,const float* b,int ldb,float (&acc)[2][4]) {
    int lane=threadIdx.x&31,row=lane/4,col=lane%4;
#pragma unroll
    for(int k=0;k<16;k+=8) {
        const int k0=EvenOdd ? ((k+col)*2)%16+(k+col)/8 : k+col, k1=EvenOdd ? ((k+col+4)*2)%16+(k+col+4)/8 : k+col+4;
        uint32_t av[4]={__float_as_uint(a[row*lda+k0])&0xffffe000u,
            __float_as_uint(a[(row+8)*lda+k0])&0xffffe000u,
            __float_as_uint(a[row*lda+k1])&0xffffe000u,
            __float_as_uint(a[(row+8)*lda+k1])&0xffffe000u};
#pragma unroll
        for(int n=0;n<2;n++) {
            uint32_t b0=__float_as_uint(b[k0*ldb+row+n*8])&0xffffe000u;
            uint32_t b1=__float_as_uint(b[k1*ldb+row+n*8])&0xffffe000u;
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[n][0]),"+f"(acc[n][1]),"+f"(acc[n][2]),"+f"(acc[n][3])
                : "r"(av[0]),"r"(av[1]),"r"(av[2]),"r"(av[3]),"r"(b0),"r"(b1));
        }
    }
}

__device__ __forceinline__ float fla_exp2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

using cs1::bf16;
using cs1::warp_sum;

constexpr int C = 64;    // chunk length
constexpr int K = 128;   // key head dim
constexpr int V = 128;   // value head dim
constexpr int THREADS = 256;
// Padded row strides for the shared matrices.
constexpr int KP = K + 4;   // packed TF32 component planes
constexpr int CP = C + 4;   // float
constexpr int HB = K + 8;   // bfloat16
constexpr int TB = C + 8;   // bfloat16

// Retain the original shared-memory reservation; these regions now hold
// normalized BF16 Q/K, then are reused for inverse/output operands.
struct Tf32Row {
    uint16_t lo[KP];
    uint8_t hi[KP];
};
static_assert(sizeof(Tf32Row) == KP * 3);



struct Work {
    bf16* u;      // [H, NCC, V]
    bf16* w;      // [H, NCC, K]
    bf16* qd;     // [H, NCC, K]  normalized Q
    bf16* kd;     // [H, NCC, K]  normalized K
    float* p;     // [H, NC, C, C] float32 scratch of q.k
    bf16* pb;     // [H, NC, C, C] (q.k) exp(cum_i - cum_j), j <= i
    float* cumulative; // [H, NCC] cumulative base-2 gates
    float* decay; // [H, NC]       exp(cum_last)
    bf16* s;      // [H, NC, K, V] state before each chunk
    bf16* vn;     // [H, NCC, V]   v_new
};

// Byte offsets of the workspace parts, 256-byte aligned.
struct Layout {
    size_t u, w, qd, kd, p, pb, decay, s, vn, cumulative, total;
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
        cumulative = take((size_t)H * NCC * 4);
        s = take((size_t)H * NC * K * V * 2);
        vn = take((size_t)H * NCC * V * 2);
        total = at;
    }
};

// kernel 1 shared memory, bytes
constexpr int TBF = C * CP * 4 + 3 * 256 * 4;  // offset of T as bf16 inside R2
constexpr int R1 = 0;                     // kn packed TF32 [C]; later v as bf16 [C][HB]
constexpr int R2 = R1 + C * sizeof(Tf32Row); // qn packed TF32; later T + scratch, then T as bf16
constexpr int R3 = R2 + TBF + C * TB * 2; // A float [C][CP]; later k exp(cum) as bf16 [C][HB]
constexpr int R4 = R3 + C * CP * 4;       // cum, beta, exp(cum), exp(cum_last - cum), k norm inverse
constexpr size_t SMEM1_BYTES = R4 + 5 * C * 4;
static_assert(C * HB * 2 <= C * CP * 4, "k exp(cum) as bf16 fits over A");
static_assert(C * sizeof(Tf32Row) <= R3 - R2, "packed qn fits in R2");
static_assert(C * HB * 2 <= R2 - R1, "v as bf16 fits over kn");

__global__ void __launch_bounds__(THREADS) gdn_chunk_prep(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
    const __nv_bfloat16* __restrict__ beta, Work ws, int T, int H, int HK, float scale) {
    extern __shared__ __align__(128) unsigned char sm[];
    Tf32Row* kn = reinterpret_cast<Tf32Row*>(sm + R1);
    Tf32Row* qn = reinterpret_cast<Tf32Row*>(sm + R2);
    float* tm = reinterpret_cast<float*>(sm + R2);
    float* sc = tm + C * CP;
    float* am = reinterpret_cast<float*>(sm + R3);
    float* cum = reinterpret_cast<float*>(sm + R4);
    float* bet = cum + C;
    float* ecum = bet + C;
    float* erem = ecum + C;
    float* kinvs = erem + C;
    __nv_bfloat16* vb = reinterpret_cast<__nv_bfloat16*>(sm + R1);
    __nv_bfloat16* tb = reinterpret_cast<__nv_bfloat16*>(sm + R2 + TBF);
    __nv_bfloat16* kw = reinterpret_cast<__nv_bfloat16*>(sm + R3);

    const int c = blockIdx.x, h = blockIdx.y, NC = gridDim.x, NCC = NC * C;
    const int hk = h / (H / HK);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // 1. cumulative decays and learning rates (padding rows stay zero)
    if (tid < C) {
        const int t = c * C + tid;
        cum[tid] = t < T ? g[(size_t)t * H + h] : 0.f;
        bet[tid] = t < T ? __bfloat162float(beta[(size_t)t * H + h]) : 0.f;
    }
    __syncthreads();
    if(tid<C) {
        float value=cum[tid];
        for(int step=1;step<32;step*=2) {
            float previous=__shfl_up_sync(0xffffffff,value,step);
            if(lane>=step) value+=previous;
        }
        cum[tid]=value;
    }
    __syncthreads();
    if(tid>=32&&tid<C) cum[tid]+=cum[31];
    __syncthreads();
    if(tid<C) cum[tid]*=1.4426950408889634f;
    __syncthreads();
    if(tid==0) ws.decay[h*NC+c]=fla_exp2(cum[C-1]);
    if(tid<C) {
        ecum[tid]=fla_exp2(cum[tid]);
        erem[tid]=fla_exp2(cum[C-1]-cum[tid]);
    }
    __syncthreads();

    // 2. L2-normalize q/k in float32 and form decayed q/k before rounding to
    //    bfloat16. Convert the pair-product operands to TF32 once per row.
    for (int base = warp*2; base < C; base += THREADS / 16) {
        const int r=base+lane/16;
        const int t = c * C + r;
        float kv[8], qv[8], ks = 0.f, qs = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = (lane%16)*8+e;
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
        for(int step=8;step>0;step/=2) {
            ks+=__shfl_xor_sync(0xffffffff,ks,step,16);
            qs+=__shfl_xor_sync(0xffffffff,qs,step,16);
        }
        const float kinv = reference_inverse_norm(ks + 1e-6f), qinv = reference_inverse_norm(qs + 1e-6f);
        if (lane%16 == 0) kinvs[r] = kinv;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = (lane%16)*8+e;
            const float kx = __bfloat162float(__float2bfloat16(kv[e] * kinv)), qx = __bfloat162float(__float2bfloat16(qv[e] * qinv));
            reinterpret_cast<bf16*>(kn)[r*HB+d]=__float2bfloat16(kx);
            reinterpret_cast<bf16*>(qn)[r*HB+d]=__float2bfloat16(qx);
            const size_t row = ((size_t)h * NCC + c * C + r) * K + d;
            ws.qd[row] = __float2bfloat16(qx);
            ws.kd[row] = __float2bfloat16(kx);
            if (d == 0) ws.cumulative[(size_t)h * NCC + c * C + r] = cum[r];
        }
    }
    __syncthreads();

    // 3. pair products on tensor cores: the 10 lower 16x16 tiles of k.k (into A) and of
    //    q.k (into P, in global memory), then the masks and decays elementwise
    float* pout = ws.p + ((size_t)h * NC + c) * C * C;
    for (int e = warp; e < 20; e += THREADS / 32) {
        const int tile = e % 10;
        const int it = tile < 1 ? 0 : tile < 3 ? 1 : tile < 6 ? 2 : 3;
        const int jt = tile - it * (it + 1) / 2;
        const bf16* asrc = reinterpret_cast<bf16*>(e < 10 ? kn : qn);
        const int group = lane / 4, thread = lane % 4;
        float acc[2][4] = {};
#pragma unroll
        for (int k0=0;k0<K;k0+=16) {
            uint32_t a[4],b[4];
            cs1::load_a(a,asrc,HB,it*16,k0,lane);
            cs1::load_b_nk(b,reinterpret_cast<bf16*>(kn),HB,k0,jt*16,lane);
            cs1::mma16816(acc[0],a,b[0],b[1]);
            cs1::mma16816(acc[1],a,b[2],b[3]);
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int row = it * 16 + group + r * 8;
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                const int col = jt * 16 + 2 * thread + nt * 8;
                float* dst = (e < 10 ? am + row * CP : pout + row * C) + col;
                *reinterpret_cast<float2*>(dst) = make_float2(acc[nt][2 * r], acc[nt][2 * r + 1]);
            }
        }
    }
    __syncthreads();
    __nv_bfloat16* pb = ws.pb + ((size_t)h * NC + c) * C * C;
    for (int x = tid; x < C * C; x += THREADS) {
        const int i = x / C, j = x % C;
        const float dec = j <= i ? fla_exp2(cum[i] - cum[j]) : 0.f;
        am[i * CP + j] = j < i ? __fmul_rn(__fmul_rn(am[i * CP + j], dec), bet[i]) : 0.f;
        pb[i * C + j] = __float2bfloat16(j <= i ? pout[i * C + j] * dec : 0.f);
    }
    __syncthreads();

    // 4. T = (I + A)^-1 in 16x16 blocks. The first reduction pair is an FMA.
    for (int x = tid; x < C * CP; x += THREADS) tm[x] = 0.f;
    __syncthreads();
    if (tid < C) {
        const int base = 16 * (tid / 16), col = tid % 16;
        float xs[16] = {};
        for(int r=1;r<16;r++) {
            float terms[16];
#pragma unroll
            for(int j=0;j<8;j++) {
                float low=j<r ? -am[(base+r)*CP+base+j] : 0.f;
                float high=j+8<r ? -am[(base+r)*CP+base+j+8] : 0.f;
                terms[j]=fmaf(low,xs[j],__fmul_rn(high,xs[j+8]));
            }
#pragma unroll
            for(int step=4;step>0;step/=2) {
#pragma unroll
                for(int j=0;j<step;j++) terms[j]=__fadd_rn(terms[j],terms[j+step]);
            }
            float direct=col<r ? -am[(base+r)*CP+base+col] : 0.f;
            xs[r]=__fadd_rn(direct,terms[0]);
        }
#pragma unroll
        for(int r=0;r<16;r++) tm[(base+r)*CP+base+col]=xs[r]+(r==col ? 1.f : 0.f);
    }
    __syncthreads();
    for(int level=1;level<4;level++) {
        int count=4-level;
        if(warp<count) {
            int bi=level+warp,bj=warp;
            float acc[2][4]={};
            if(level==1) reference_dot16<true>(tm+16*bi*CP+16*bi,CP,am+16*bi*CP+16*bj,CP,acc);
            else for(int kb=bj;kb<bi;kb++) {
                reference_dot16<true>(am+16*bi*CP+16*kb,CP,tm+16*kb*CP+16*bj,CP,acc);
            }
            int row=lane/4,col=2*(lane%4);
#pragma unroll
            for(int n=0;n<2;n++) for(int e=0;e<4;e++) sc[warp*256+(row+(e/2)*8)*16+col+n*8+e%2]=acc[n][e];
        }
        __syncthreads();
        if(warp<count) {
            int bi=level+warp,bj=warp;
            float acc[2][4]={};
            if(level==1) reference_dot16<false>(sc+warp*256,16,tm+16*bj*CP+16*bj,CP,acc);
            else reference_dot16<false>(tm+16*bi*CP+16*bi,CP,sc+warp*256,16,acc);
            int row=lane/4,col=2*(lane%4);
#pragma unroll
            for(int n=0;n<2;n++) for(int e=0;e<4;e++) tm[(bi*16+row+(e/2)*8)*CP+bj*16+col+n*8+e%2]=-acc[n][e];
        }
        __syncthreads();
    }

    // 5. bfloat16 operands: k exp(cum) over A, T beta after T, then v over kn;
    //    u = T (beta v) and w = T (beta exp(cum) k) on tensor cores, stored as bfloat16
    for (int x = tid; x < C * K; x += THREADS) {
        const int j = x / K, d = x % K;
        const int t = c * C + j;
        const float kx = t < T ? __bfloat162float(k[((size_t)t * HK + hk) * K + d]) : 0.f;
        const float normalized = __bfloat162float(__float2bfloat16(kx * kinvs[j]));
        kw[j * HB + d] = __float2bfloat16(__bfloat162float(__float2bfloat16(normalized * bet[j])) * ecum[j]);
    }
    for (int x = tid; x < C * C; x += THREADS) {
        const int i = x / C, j = x % C;
        tb[i * TB + j] = __float2bfloat16(tm[i * CP + j]);

    }
    __syncthreads();
    for (int x = tid; x < C * V; x += THREADS) {
        const int j = x / V, d = x % V;
        const int t = c * C + j;
        vb[j * HB + d] = t < T ? __float2bfloat16(__bfloat162float(v[((size_t)t * H + h) * V + d]) * bet[j]) : __float2bfloat16(0.f);
    }
    __syncthreads();
    for (int e = warp; e < 64; e += THREADS / 32) {
        const bool is_u = e < 32;
        const int it = (e % 32) / 8, dt = e % 8;
        const __nv_bfloat16* bsrc = is_u ? vb : kw;
        float acc[2][4] = {};
        for (int kb = 0; kb <= it; kb++) {  // T is lower triangular
            uint32_t a[4], b[4];
            cs1::load_a(a, tb, TB, it * 16, kb * 16, lane);
            cs1::load_b_kn(b, bsrc, HB, kb * 16, dt * 16, lane);
            cs1::mma16816(acc[0], a, b[0], b[1]);
            cs1::mma16816(acc[1], a, b[2], b[3]);
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int row = it * 16 + lane / 4 + r * 8;
            bf16* dst = (is_u ? ws.u : ws.w) + ((size_t)h * NCC + c * C + row) * K + dt * 16 + 2 * (lane % 4);
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
                *reinterpret_cast<uint32_t*>(dst + nt * 8) = cs1::pack_bf16(acc[nt][2 * r], acc[nt][2 * r + 1]);
        }
    }
}

// ---- kernel 2: the state, chunk by chunk ----

constexpr int BVS = 32;             // value columns per block
constexpr int ST_THREADS = 128;
constexpr int WS_LD = K + 8;        // bfloat16 row stride of staged w and kd
constexpr int SS_LD = BVS + 8;      // bfloat16 row stride of the S copy and v_new
constexpr int STAGE = C * WS_LD;    // elements of one staged w or kd
constexpr size_t SMEM2_BYTES = (4 * STAGE + K * SS_LD + C * SS_LD) * 2;

__global__ void __launch_bounds__(ST_THREADS) gdn_chunk_state(Work ws, int NC, const float* __restrict__ S_IN,
                                                              float* __restrict__ S_OUT) {
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
    if (S_IN != nullptr) {
        // float32 state seeded for a continuation: identical to the registers a
        // full pass would carry into this chunk, so the scan repeats one pass.
        const float* si = S_IN + (size_t)h * K * V;
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int row = warp * 32 + mt * 16 + g + (e >> 1) * 8;
                    const int col = vb0 + nt * 8 + 2 * t + (e & 1);
                    st[mt][nt][e] = si[(size_t)row * V + col];
                }
    }

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
                const float x = __low2float(uv) - acc[nt][2 * r], y = __high2float(uv) - acc[nt][2 * r + 1];
                const float erem = fla_exp2(ws.cumulative[(size_t)h*NCC + c*C + C-1] - ws.cumulative[row0 + row]);
                const uint32_t pk = cs1::pack_bf16(x, y);
                *reinterpret_cast<uint32_t*>(vnew + (warp * 16 + row) * SS_LD + col) = cs1::pack_bf16(x * erem, y * erem);
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
    if (S_OUT != nullptr) {
        float* so = S_OUT + (size_t)h * K * V;
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int row = warp * 32 + mt * 16 + g + r * 8;
                    const int col = vb0 + nt * 8 + 2 * t;
                    *reinterpret_cast<float2*>(so + (size_t)row * V + col) =
                        make_float2(st[mt][nt][2 * r], st[mt][nt][2 * r + 1]);
                }
    }
}

// ---- kernel 3: the output, per chunk ----

constexpr int OUT_THREADS = 128;
constexpr int O_LD = K + 8;  // bfloat16 row stride of qd, S and v_new (all 128 wide)
constexpr int P_LD = C + 8;  // bfloat16 row stride of P
constexpr size_t SMEM3_BYTES = (C * O_LD + K * O_LD + C * P_LD + C * O_LD) * 2;
static_assert(K == V, "S, qd and v_new share a row stride");

__global__ void __launch_bounds__(OUT_THREADS) gdn_chunk_out(Work ws, bf16* __restrict__ o, int T, int H, float scale) {
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
    float intra[V / 8][4] = {};
    // P v_new; P is lower triangular, so these rows need positions below (warp + 1) * 16
    for (int kk = 0; kk < (warp + 1) * 16; kk += 16) {
        uint32_t a[4];
        cs1::load_a(a, ps, P_LD, warp * 16, kk, lane);
#pragma unroll
        for (int nt = 0; nt < V / 8; nt += 2) {
            uint32_t b[4];
            cs1::load_b_kn(b, vs, O_LD, kk, nt * 8, lane);
            cs1::mma16816(intra[nt], a, b[0], b[1]);
            cs1::mma16816(intra[nt + 1], a, b[2], b[3]);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int tok = c * C + warp * 16 + g + r * 8;
        if (tok >= T) continue;
        bf16* dst = o + ((size_t)tok * H + h) * V + 2 * t;
#pragma unroll
        for (int nt = 0; nt < V / 8; nt++)
            *reinterpret_cast<uint32_t*>(dst + nt * 8) = cs1::pack_bf16(fmaf(intra[nt][2*r], scale, __fmul_rn(__fmul_rn(acc[nt][2*r], fla_exp2(ws.cumulative[rows + warp*16+g+r*8])), scale)), fmaf(intra[nt][2*r+1], scale, __fmul_rn(__fmul_rn(acc[nt][2*r+1], fla_exp2(ws.cumulative[rows + warp*16+g+r*8])), scale)));
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
    ws.cumulative = reinterpret_cast<float*>(b + l.cumulative);
    ws.s = reinterpret_cast<bf16*>(b + l.s);
    ws.vn = reinterpret_cast<bf16*>(b + l.vn);
    return ws;
}

}  // namespace

extern "C" {

size_t cs1_reference_gdn_workspace_floats(int T, int H) { return (Layout(T, H).total + 3) / 4; }

static int gdn_prefill_run(const void* q, const void* k, const void* v, const float* g,
                           const void* beta, void* o, float* workspace, int T, int H, int HK,
                           float scale, const void* s_in, void* s_out, void* stream) {
    if (T < 0 || HK <= 0 || H % HK != 0) return cudaErrorInvalidValue;
    const float* si = static_cast<const float*>(s_in);
    float* so = static_cast<float*>(s_out);
    if (T == 0) {
        // No tokens: the state passes through unchanged, for symmetric capture.
        if (so == nullptr) return cudaSuccess;
        const size_t bytes = (size_t)H * K * V * sizeof(float);
        cudaStream_t st = static_cast<cudaStream_t>(stream);
        return si != nullptr ? cudaMemcpyAsync(so, si, bytes, cudaMemcpyDeviceToDevice, st)
                             : cudaMemsetAsync(so, 0, bytes, st);
    }
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
    gdn_chunk_state<<<dim3(H, V / BVS), ST_THREADS, SMEM2_BYTES, st>>>(ws, NC, si, so);
    gdn_chunk_out<<<dim3(NC, H), OUT_THREADS, SMEM3_BYTES, st>>>(ws, static_cast<bf16*>(o), T, H, scale);
    return cudaGetLastError();
}

int cs1_reference_gdn_prefill(const void* q, const void* k, const void* v, const float* g, const void* beta,
                    void* o, float* workspace, int T, int H, int HK, float scale, void* stream) {
    return gdn_prefill_run(q, k, v, g, beta, o, workspace, T, H, HK, scale, nullptr, nullptr, stream);
}

int cs1_reference_gdn_prefill_x(const void* q, const void* k, const void* v, const float* g, const void* beta,
                      void* o, float* workspace, int T, int H, int HK, float scale,
                      const void* s_in, void* s_out, void* stream) {
    return gdn_prefill_run(q, k, v, g, beta, o, workspace, T, H, HK, scale, s_in, s_out, stream);
}

}  // extern "C"
