// Helpers shared by the Qwen3.5 kernels.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cs1 {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }
// A float rounded through bfloat16, as PyTorch stores the result of each bfloat16 op.
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

// Sum over the block; `scratch` holds at least 32 floats. Every thread gets the sum.
__device__ __forceinline__ float block_sum(float x, float* scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    x = warp_sum(x);
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        float t = lane < warps ? scratch[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) scratch[0] = t;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// PyTorch's float32 SiLU and sigmoid on CUDA.
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// 8 bfloat16 values, one 16-byte load or store.
struct alignas(16) Pack8 {
    bf16 v[8];
};

__device__ __forceinline__ void load8(const bf16* p, float out[8]) {
    const Pack8 pk = *reinterpret_cast<const Pack8*>(p);
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = f32(pk.v[i]);
}

__device__ __forceinline__ void store8(bf16* p, const float in[8]) {
    Pack8 pk;
#pragma unroll
    for (int i = 0; i < 8; i++) pk.v[i] = to_bf16(in[i]);
    *reinterpret_cast<Pack8*>(p) = pk;
}

}  // namespace cs1
