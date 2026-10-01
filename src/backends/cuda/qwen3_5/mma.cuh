// Tensor-core building blocks for sm_80 and later: mma.sync m16n8k16 on bfloat16 with
// float32 accumulation, ldmatrix, and cp.async.
//
// Fragment layouts (PTX ISA, mma.m16n8k16): with g = lane / 4 and t = lane % 4, an
// accumulator holds rows g (elements 0, 1) and g + 8 (elements 2, 3) at columns 2t and
// 2t + 1 of its 8-column tile; the A operand registers are (row g, cols 2t..),
// (row g + 8, cols 2t..), (row g, cols 8 + 2t..) and (row g + 8, cols 8 + 2t..).
#pragma once

#include "common.cuh"

namespace cs1 {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte asynchronous copy from global to shared memory; zero-fills when !valid.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid = true) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const bf16* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const bf16* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

// A fragment of the 16x16 tile at (row0, col0) of a row-major shared matrix.
__device__ __forceinline__ void load_a(uint32_t (&a)[4], const bf16* m, int ld, int row0, int col0, int lane) {
    ldmatrix_x4(a, m + (row0 + (lane % 8) + ((lane / 8) % 2) * 8) * ld + col0 + (lane / 16) * 8);
}
// A fragment of the 16x16 tile at (row0, col0) of the transpose of a row-major shared
// matrix: rows of A are columns of m.
__device__ __forceinline__ void load_a_trans(uint32_t (&a)[4], const bf16* m, int ld, int row0, int col0,
                                             int lane) {
    ldmatrix_x4_trans(a, m + (col0 + (lane % 8) + (lane / 16) * 8) * ld + row0 + ((lane / 8) % 2) * 8);
}
// B fragments of two 16x8 tiles (k0.., n0..) and (k0.., n0 + 8..) of a row-major
// shared matrix whose rows are k: b[0], b[1] and b[2], b[3].
__device__ __forceinline__ void load_b_kn(uint32_t (&b)[4], const bf16* m, int ld, int k0, int n0, int lane) {
    ldmatrix_x4_trans(b, m + (k0 + (lane % 8) + ((lane / 8) % 2) * 8) * ld + n0 + (lane / 16) * 8);
}
// The same when the shared matrix is stored with rows n (each row contiguous in k).
__device__ __forceinline__ void load_b_nk(uint32_t (&b)[4], const bf16* m, int ld, int k0, int n0, int lane) {
    ldmatrix_x4(b, m + (n0 + (lane % 8) + (lane / 16) * 8) * ld + k0 + ((lane / 8) % 2) * 8);
}

// d += a * b for a 16x16 (row-major) by 16x8 (column-major) tile.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

}  // namespace cs1
