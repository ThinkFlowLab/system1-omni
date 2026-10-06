// Embedding lookup, the Gated DeltaNet conv and gates, and the elementwise
// activations of Qwen3.5.
#include "common.cuh"
#include "ops.h"

namespace cs1 {
namespace {

constexpr int THREADS = 256;

__global__ void embed_kernel(const int32_t* __restrict__ ids, const Pack8* __restrict__ table,
                             Pack8* __restrict__ out, int packs) {
    const size_t t = blockIdx.x;
    const size_t id = ids[t];
    for (int i = threadIdx.x; i < packs; i += blockDim.x) out[t * packs + i] = table[id * packs + i];
}

// F.conv1d in bfloat16 (float32 accumulation, rounded), then SiLU (rounded again),
// written to three contiguous outputs. Positions before the first row come from
// history [3, channels] when there is one; without it they are skipped, as the zero
// padding of a sequence's start adds nothing.
__global__ void gdn_conv_kernel(const bf16* __restrict__ qkv, int ld, const bf16* __restrict__ w,
                                const bf16* __restrict__ history, bf16* __restrict__ q,
                                bf16* __restrict__ k, bf16* __restrict__ v, int T, int key_dim,
                                int value_dim) {
    const int channels = 2 * key_dim + value_dim;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * channels) return;
    const int t = idx / channels, c = idx % channels;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int s = t - 3 + j;
        if (s >= 0)
            acc = fmaf(f32(w[c * 4 + j]), f32(qkv[(size_t)s * ld + c]), acc);
        else if (history)
            acc = fmaf(f32(w[c * 4 + j]), f32(history[(size_t)(3 + s) * channels + c]), acc);
    }
    const bf16 y = to_bf16(silu(round_bf16(acc)));
    if (c < key_dim)
        q[(size_t)t * key_dim + c] = y;
    else if (c < 2 * key_dim)
        k[(size_t)t * key_dim + c - key_dim] = y;
    else
        v[(size_t)t * value_dim + c - 2 * key_dim] = y;
}

// The conv inputs of the last three positions, from qkv or, before its first row, from
// history (zeros without one).
__global__ void gdn_conv_history_kernel(const bf16* __restrict__ qkv, int ld, const bf16* __restrict__ history,
                                        bf16* __restrict__ out, int T, int channels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3 * channels) return;
    const int r = idx / channels, c = idx % channels, s = T - 3 + r;
    out[idx] = s >= 0 ? qkv[(size_t)s * ld + c] : history ? history[(3 + s) * channels + c] : to_bf16(0.f);
}

// beta = sigmoid(b) in bfloat16; g = -exp(A_log) * softplus(a + dt_bias) in float32
// (F.softplus with threshold 20).
__global__ void gdn_gates_kernel(const bf16* __restrict__ b, const bf16* __restrict__ a, int ld,
                                 const bf16* __restrict__ A_log, const bf16* __restrict__ dt_bias,
                                 bf16* __restrict__ beta, float* __restrict__ g, int n, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int h = i % H;
    const size_t src = (size_t)(i / H) * ld + h;
    beta[i] = to_bf16(sigmoid(f32(b[src])));
    const float x = f32(a[src]) + f32(dt_bias[h]);
    const float sp = x > 20.f ? x : log1pf(expf(x));
    g[i] = -expf(f32(A_log[h])) * sp;
}

// attn_output * torch.sigmoid(gate), both bfloat16.
__global__ void sigmoid_gate_kernel(bf16* __restrict__ x, const bf16* __restrict__ gate, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = to_bf16(f32(x[i]) * round_bf16(sigmoid(f32(gate[i]))));
}

// act_fn(gate_proj(x)) * up_proj(x), both bfloat16; gate and up are the two halves of
// each row of gate_up.
__global__ void silu_mul_kernel(const bf16* __restrict__ gate_up, int ld, bf16* __restrict__ out, int I,
                                size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = i / I, j = i % I;
    const bf16* row = gate_up + t * ld;
    out[i] = to_bf16(round_bf16(silu(f32(row[j]))) * f32(row[I + j]));
}

__global__ void silu_mul_packed_kernel(const Pack8* __restrict__ gate_up, int ld,
                                       Pack8* __restrict__ out, int I, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = i / I, j = i % I;
    const Pack8* row = gate_up + t * ld;
    const Pack8 gate = row[j], up = row[I + j];
    Pack8 result;
#pragma unroll
    for (int e = 0; e < 8; e++)
        result.v[e] = to_bf16(round_bf16(silu(f32(gate.v[e]))) * f32(up.v[e]));
    out[i] = result;
}

unsigned blocks(size_t n) { return (unsigned)((n + THREADS - 1) / THREADS); }

}  // namespace
}  // namespace cs1

using namespace cs1;

extern "C" int cs1_embed(const int32_t* ids, const void* table, void* out, int T, int D, void* stream) {
    if (D % 8 != 0) return cudaErrorInvalidValue;
    if (T <= 0) return cudaSuccess;
    embed_kernel<<<T, THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        ids, static_cast<const Pack8*>(table), static_cast<Pack8*>(out), D / 8);
    return cudaGetLastError();
}

extern "C" int cs1_gdn_conv_history(const void* qkv, int ld, const void* w, const void* history, void* history_out,
                                    void* q, void* k, void* v, int T, int key_dim, int value_dim, void* stream) {
    if (T < 0 || key_dim < 0 || value_dim < 0 || ld < 2 * key_dim + value_dim) return cudaErrorInvalidValue;
    const int channels = 2 * key_dim + value_dim;
    const size_t n = (size_t)T * channels;
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (n > 0) {
        gdn_conv_kernel<<<blocks(n), THREADS, 0, st>>>(
            static_cast<const bf16*>(qkv), ld, static_cast<const bf16*>(w), static_cast<const bf16*>(history),
            static_cast<bf16*>(q), static_cast<bf16*>(k), static_cast<bf16*>(v), T, key_dim, value_dim);
    }
    if (history_out && channels > 0) {
        gdn_conv_history_kernel<<<blocks((size_t)3 * channels), THREADS, 0, st>>>(
            static_cast<const bf16*>(qkv), ld, static_cast<const bf16*>(history), static_cast<bf16*>(history_out),
            T, channels);
    }
    return cudaGetLastError();
}

extern "C" int cs1_gdn_conv(const void* qkv, int ld, const void* w, void* q, void* k, void* v, int T, int key_dim,
                            int value_dim, void* stream) {
    return cs1_gdn_conv_history(qkv, ld, w, nullptr, nullptr, q, k, v, T, key_dim, value_dim, stream);
}

extern "C" int cs1_gdn_gates(const void* b, const void* a, int ld, const void* A_log, const void* dt_bias,
                             void* beta, float* g, int T, int H, void* stream) {
    if (T < 0 || H < 0 || ld < H) return cudaErrorInvalidValue;
    const int n = T * H;
    if (n == 0) return cudaSuccess;
    gdn_gates_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const bf16*>(b), static_cast<const bf16*>(a), ld, static_cast<const bf16*>(A_log),
        static_cast<const bf16*>(dt_bias), static_cast<bf16*>(beta), g, n, H);
    return cudaGetLastError();
}

extern "C" int cs1_sigmoid_gate(void* x, const void* gate, size_t n, void* stream) {
    if (n == 0) return cudaSuccess;
    sigmoid_gate_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<bf16*>(x), static_cast<const bf16*>(gate), n);
    return cudaGetLastError();
}

extern "C" int cs1_silu_mul(const void* gate_up, int ld, void* out, int T, int I, void* stream) {
    if (T < 0 || I < 0 || ld < 2 * I) return cudaErrorInvalidValue;
    const size_t n = (size_t)T * I;
    if (n == 0) return cudaSuccess;
    if (I % 8 == 0 && ld % 8 == 0 &&
        ((reinterpret_cast<uintptr_t>(gate_up) | reinterpret_cast<uintptr_t>(out)) & 15) == 0) {
        silu_mul_packed_kernel<<<blocks(n / 8), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const Pack8*>(gate_up), ld / 8, static_cast<Pack8*>(out), I / 8, n / 8);
    } else {
        silu_mul_kernel<<<blocks(n), THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const bf16*>(gate_up), ld, static_cast<bf16*>(out), I, n);
    }
    return cudaGetLastError();
}
