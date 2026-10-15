// FP32 chunked Gated DeltaNet for Transformers' no-FLA text reference.
// Normalize/scale before dot products; solve the unit lower-triangular systems
// in FP32; keep every recurrent-state operation FP32 until the BF16 output.
#include "common.cuh"
#include "ops.h"
#include <cmath>
#include <cublas_v2.h>
#include <new>

namespace {
constexpr int C = 64, D = 128;
struct Context {
  cublasHandle_t blas = nullptr;
  void *scratch = nullptr;
  size_t bytes = 0;
  cudaStream_t stream = nullptr;
  int device = 0;
};
struct Layout {
  size_t q, k, kbeta, qd, kd, u, w, a, p, cum, state, vnew, temp, out, plus,
      next, pa, pu, pw, total;
  Layout(int tokens, int heads) {
    size_t nc = (tokens + C - 1) / C, ncc = nc * C, at = 0;
    auto take = [&](size_t bytes) {
      size_t offset = at;
      at = (at + bytes + 255) / 256 * 256;
      return offset;
    };
    size_t vector = (size_t)heads * ncc * D * 4,
           matrix = (size_t)heads * nc * C * C * 4;
    q = take(vector);
    k = take(vector);
    kbeta = take(vector);
    qd = take(vector);
    kd = take(vector);
    u = take(vector);
    w = take(vector);
    a = take(matrix);
    p = take(matrix);
    cum = take((size_t)heads * ncc * 4);
    state = take((size_t)heads * D * D * 4);
    vnew = take((size_t)heads * C * D * 4);
    temp = take((size_t)heads * C * D * 4);
    out = take((size_t)heads * C * D * 4);
    plus = take((size_t)heads * C * D * 4);
    next = take((size_t)heads * D * D * 4);
    pa = take((size_t)heads * nc * sizeof(void *));
    pu = take((size_t)heads * nc * sizeof(void *));
    pw = take((size_t)heads * nc * sizeof(void *));
    total = at;
  }
};
struct Work {
  float *q, *k, *kbeta, *qd, *kd, *u, *w, *a, *p, *cum, *state, *vnew, *temp,
      *out, *plus, *next;
  const float **pa;
  float **pu;
  float **pw;
};
Work bind(float *workspace, const Layout &l) {
  auto p = [&](size_t offset) { return (float *)((char *)workspace + offset); };
  return {p(l.q),
          p(l.k),
          p(l.kbeta),
          p(l.qd),
          p(l.kd),
          p(l.u),
          p(l.w),
          p(l.a),
          p(l.p),
          p(l.cum),
          p(l.state),
          p(l.vnew),
          p(l.temp),
          p(l.out),
          p(l.plus),
          p(l.next),
          (const float **)p(l.pa),
          (float **)p(l.pu),
          (float **)p(l.pw)};
}
// Association order: Torch v2.14.0 aten/native/cuda/ScanUtils.cuh,
// scan_innermost_dim (not a sequential recurrence).
// Match the reference's row-dependent Sklansky scan, including the carry
// inserted into the first element when 64 values are scanned in two blocks.
__global__ void cumulative(const float *g, float *out, int tokens, int heads,
                           int nc, int width) {
  __shared__ float values[C];
  int b = blockIdx.x, head = b / nc, chunk = b % nc, lane = threadIdx.x;
  int count = width == 16 ? 32 : 64;
  float carry = 0.f;
  for (int base = 0; base < C; base += count) {
    for (int i = lane; i < count; i += 32) {
      int token = chunk * C + base + i;
      values[i] = token < tokens ? g[(size_t)token * heads + head] : 0.f;
    }
    __syncthreads();
    if (lane == 0)
      values[0] = __fadd_rn(values[0], carry);
    __syncthreads();
    for (int step = 1; step < count; step *= 2) {
      if (lane < count / 2) {
        int a = (lane / step) * (2 * step) + step;
        int ti = a + lane % step;
        values[ti] = __fadd_rn(values[ti], values[a - 1]);
      }
      __syncthreads();
    }
    for (int i = lane; i < count; i += 32)
      out[(size_t)b * C + base + i] = values[i];
    carry = values[count - 1];
    __syncthreads();
  }
}
__global__ void prepare(const cs1::bf16 *q, const cs1::bf16 *k,
                        const cs1::bf16 *v, const cs1::bf16 *beta, Work ws,
                        int tokens, int heads, int key_heads, int nc,
                        float scale) {
  int head = blockIdx.x / (nc * C), row = blockIdx.x % (nc * C),
      chunk = row / C, r = row % C, lane = threadIdx.x;
  int kh = head / (heads / key_heads), token = chunk * C + r;
  float qv[4], kv[4], qs[4] = {}, ks[4] = {};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int d = lane * 4 + j;
    qv[j] = token < tokens
                ? cs1::f32(q[((size_t)token * key_heads + kh) * D + d])
                : 0.f;
    kv[j] = token < tokens
                ? cs1::f32(k[((size_t)token * key_heads + kh) * D + d])
                : 0.f;
    qs[j] = qv[j] * qv[j];
    ks[j] = kv[j] * kv[j];
  }
  float qsum = ((qs[0] + qs[1]) + qs[2]) + qs[3],
        ksum = ((ks[0] + ks[1]) + ks[2]) + ks[3];
  for (int step = 16; step > 0; step /= 2) {
    qsum += __shfl_down_sync(0xffffffff, qsum, step);
    ksum += __shfl_down_sync(0xffffffff, ksum, step);
  }
  float qi = rsqrtf(__shfl_sync(0xffffffff, qsum, 0) + 1e-6f),
        ki = rsqrtf(__shfl_sync(0xffffffff, ksum, 0) + 1e-6f);
  float cum = ws.cum[(size_t)head * nc * C + row],
        last = ws.cum[((size_t)head * nc + chunk) * C + C - 1];
  float b = token < tokens ? cs1::f32(beta[(size_t)token * heads + head]) : 0.f;
  float ec = expf(cum), er = expf(last - cum);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int d = lane * 4 + j;
    size_t index = ((size_t)head * nc * C + row) * D + d;
    float qn = __fmul_rn(__fmul_rn(qv[j], qi), scale),
          kn = __fmul_rn(kv[j], ki), kb = __fmul_rn(kn, b);
    ws.q[index] = qn;
    ws.k[index] = kn;
    ws.kbeta[index] = kb;
    ws.qd[index] = __fmul_rn(qn, ec);
    ws.kd[index] = __fmul_rn(kn, er);
    ws.u[index] =
        token < tokens
            ? __fmul_rn(cs1::f32(v[((size_t)token * heads + head) * D + d]), b)
            : 0.f;
    ws.w[index] = __fmul_rn(kb, ec);
  }
}
__global__ void mask(Work ws, int batches) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x,
         total = (size_t)batches * C * C;
  if (index >= total)
    return;
  int i = (index % (C * C)) / C, j = index % C;
  size_t batch = index / (C * C);
  float decay =
      j <= i ? expf(ws.cum[batch * C + i] - ws.cum[batch * C + j]) : 0.f;
  ws.a[index] = __fmul_rn(ws.a[index], decay);
  ws.p[index] = __fmul_rn(ws.p[index], decay);
}
__global__ void pointers(Work ws, int batches) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batches)
    return;
  ws.pa[b] = ws.a + (size_t)b * C * C;
  ws.pu[b] = ws.u + (size_t)b * C * D;
  ws.pw[b] = ws.w + (size_t)b * C * D;
}
__global__ void subtract(Work ws, int heads, int nc, int chunk) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x,
         total = (size_t)heads * C * D;
  if (i >= total)
    return;
  size_t h = i / (C * D), r = i % (C * D);
  ws.vnew[i] = __fsub_rn(ws.u[(h * nc + chunk) * C * D + r], ws.temp[i]);
}
__global__ void output(Work ws, cs1::bf16 *out, int tokens, int heads,
                       int chunk) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x,
         total = (size_t)heads * C * D;
  if (i >= total)
    return;
  int head = i / (C * D), row = (i % (C * D)) / D, d = i % D,
      token = chunk * C + row;
  if (token < tokens)
    out[((size_t)token * heads + head) * D + d] =
        cs1::to_bf16(__fadd_rn(ws.out[i], ws.plus[i]));
}
__global__ void update(Work ws, int heads, int nc, int chunk) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x,
         total = (size_t)heads * D * D;
  if (i >= total)
    return;
  size_t h = i / (D * D);
  float decay = expf(ws.cum[(h * nc + chunk) * C + C - 1]);
  ws.state[i] = __fadd_rn(__fmul_rn(ws.state[i], decay), ws.next[i]);
}
int status(cublasStatus_t s) {
  return s == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + (int)s;
}
} // namespace
extern "C" void *cs1_gdn_fp32_create(size_t bytes) {
  Context *c = new (std::nothrow) Context;
  if (!c)
    return nullptr;
  c->bytes = bytes;
  if (cudaGetDevice(&c->device) != cudaSuccess ||
      cublasCreate(&c->blas) != CUBLAS_STATUS_SUCCESS) {
    delete c;
    return nullptr;
  }
  if (bytes && cudaMalloc(&c->scratch, bytes) != cudaSuccess) {
    cublasDestroy(c->blas);
    delete c;
    return nullptr;
  }
  if (cublasSetMathMode(c->blas, CUBLAS_DEFAULT_MATH) !=
      CUBLAS_STATUS_SUCCESS) {
    if (c->scratch)
      cudaFree(c->scratch);
    cublasDestroy(c->blas);
    delete c;
    return nullptr;
  }
  return c;
}
extern "C" void cs1_gdn_fp32_destroy(void *opaque) {
  auto *c = (Context *)opaque;
  if (!c)
    return;
  int previous = 0;
  cudaGetDevice(&previous);
  cudaSetDevice(c->device);
  cudaStreamSynchronize(c->stream);
  cublasDestroy(c->blas);
  if (c->scratch)
    cudaFree(c->scratch);
  delete c;
  cudaSetDevice(previous);
}
extern "C" size_t cs1_gdn_fp32_workspace_floats(int tokens, int heads) {
  if (tokens <= 0 || tokens > 65536 || heads <= 0 || heads > 256)
    return 0;
  return Layout(tokens, heads).total / 4;
}
extern "C" int cs1_gdn_prefill_fp32(void *opaque, const void *q, const void *k,
                                    const void *v, const float *g,
                                    const void *beta, void *out,
                                    float *workspace, int tokens, int heads,
                                    int key_heads, float scale, void *stream) {
  auto *c = (Context *)opaque;
  if (!c || !workspace || !q || !k || !v || !g || !beta || !out ||
      !cs1_gdn_fp32_workspace_floats(tokens, heads) || key_heads <= 0 ||
      heads % key_heads != 0 || !std::isfinite(scale) || scale <= 0.f)
    return cudaErrorInvalidValue;
  int nc = (tokens + C - 1) / C, batches = heads * nc;
  c->stream = (cudaStream_t)stream;
  Work ws = bind(workspace, Layout(tokens, heads));
  int rc = status(cublasSetStream(c->blas, c->stream));
  if (rc)
    return rc;
  if (c->bytes) {
    rc = status(cublasSetWorkspace(c->blas, c->scratch, c->bytes));
    if (rc)
      return rc;
  }
  int log_rows = 0;
  while ((1 << log_rows) < batches)
    ++log_rows;
  int log_width = (9 + 6 - log_rows) / 2;
  int width = 1 << max(4, min(9, log_width));
  cumulative<<<batches, 32, 0, c->stream>>>(g, ws.cum, tokens, heads, nc,
                                            width);
  prepare<<<heads * nc * C, 32, 0, c->stream>>>(
      (const cs1::bf16 *)q, (const cs1::bf16 *)k, (const cs1::bf16 *)v,
      (const cs1::bf16 *)beta, ws, tokens, heads, key_heads, nc, scale);
  rc = cudaGetLastError();
  if (rc)
    return rc;
  const float one = 1.f, zero = 0.f;
#define BLAS(expr)                                                             \
  do {                                                                         \
    rc = status(expr);                                                         \
    if (rc)                                                                    \
      return rc;                                                               \
  } while (0)
  BLAS(cublasSgemmStridedBatched(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, C, C, D,
                                 &one, ws.k, D, C * D, ws.kbeta, D, C * D,
                                 &zero, ws.a, C, C * C, batches));
  BLAS(cublasSgemmStridedBatched(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, C, C, D,
                                 &one, ws.k, D, C * D, ws.q, D, C * D, &zero,
                                 ws.p, C, C * C, batches));
  mask<<<((size_t)batches * C * C + 255) / 256, 256, 0, c->stream>>>(ws,
                                                                     batches);
  pointers<<<(batches + 127) / 128, 128, 0, c->stream>>>(ws, batches);
  rc = cudaGetLastError();
  if (rc)
    return rc;
  BLAS(cublasStrsmBatched(c->blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                          CUBLAS_OP_N, CUBLAS_DIAG_UNIT, D, C, &one, ws.pa, C,
                          ws.pu, D, batches));
  BLAS(cublasStrsmBatched(c->blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                          CUBLAS_OP_N, CUBLAS_DIAG_UNIT, D, C, &one, ws.pa, C,
                          ws.pw, D, batches));
  rc = cudaMemsetAsync(ws.state, 0, (size_t)heads * D * D * 4, c->stream);
  if (rc)
    return rc;
  for (int chunk = 0; chunk < nc; chunk++) {
    size_t offset = (size_t)chunk * C * D;
    long long stride = (long long)nc * C * D;
    BLAS(cublasSgemmStridedBatched(c->blas, CUBLAS_OP_N, CUBLAS_OP_N, D, C, D,
                                   &one, ws.state, D, D * D, ws.w + offset, D,
                                   stride, &zero, ws.temp, D, C * D, heads));
    subtract<<<((size_t)heads * C * D + 255) / 256, 256, 0, c->stream>>>(
        ws, heads, nc, chunk);
    BLAS(cublasSgemmStridedBatched(c->blas, CUBLAS_OP_N, CUBLAS_OP_N, D, C, D,
                                   &one, ws.state, D, D * D, ws.qd + offset, D,
                                   stride, &zero, ws.out, D, C * D, heads));
    BLAS(cublasSgemmStridedBatched(
        c->blas, CUBLAS_OP_N, CUBLAS_OP_N, D, C, C, &one, ws.vnew, D, C * D,
        ws.p + (size_t)chunk * C * C, C, (long long)nc * C * C, &zero, ws.plus,
        D, C * D, heads));
    output<<<((size_t)heads * C * D + 255) / 256, 256, 0, c->stream>>>(
        ws, (cs1::bf16 *)out, tokens, heads, chunk);
    BLAS(cublasSgemmStridedBatched(c->blas, CUBLAS_OP_N, CUBLAS_OP_T, D, D, C,
                                   &one, ws.vnew, D, C * D, ws.kd + offset, D,
                                   stride, &zero, ws.next, D, D * D, heads));
    update<<<((size_t)heads * D * D + 255) / 256, 256, 0, c->stream>>>(
        ws, heads, nc, chunk);
    rc = cudaGetLastError();
    if (rc)
      return rc;
  }
#undef BLAS
  return cudaSuccess;
}
