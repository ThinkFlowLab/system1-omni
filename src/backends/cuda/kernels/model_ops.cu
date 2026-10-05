// Glue operations around the exported official TileLang encoder.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>
using BF = __nv_bfloat16;

// Reduction order follows PyTorch 2.11 CUDA LayerNorm (BSD-3-Clause).
// See THIRD_PARTY.md. Four warps, four adjacent values per vector, then tree reduction.
struct Stats {
  float mean, var, count;
};

__device__ Stats combine(Stats b, Stats a) {
  float delta = b.mean - a.mean, count = a.count + b.count;
  if (count > 0) {
    float coef = 1.f / count, na = a.count * coef, nb = b.count * coef;
    return {na * a.mean + nb * b.mean,
            a.var + b.var + delta * delta * a.count * nb, count};
  }
  return {0, 0, 0};
}

template <class Load>
__device__ Stats stats(Load load, float* buf) {
  int lane = threadIdx.x, warp = threadIdx.y, t = lane + warp * 32;
  Stats wd{0, 0, 0};
  for (int i = t; i < 256; i += 128) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      float v = load(4 * i + j), delta = v - wd.mean, count = wd.count + 1.f,
            mean = wd.mean + delta * (1.f / count);
      wd = {mean, wd.var + delta * (v - mean), count};
    }
  }
  for (int offset = 16; offset; offset >>= 1) {
    Stats other{__shfl_down_sync(0xffffffff, wd.mean, offset),
                __shfl_down_sync(0xffffffff, wd.var, offset),
                __shfl_down_sync(0xffffffff, wd.count, offset)};
    wd = combine(wd, other);
  }
  for (int offset = 2; offset; offset >>= 1) {
    if (lane == 0 && warp >= offset && warp < 2 * offset) {
      int j = warp - offset;
      buf[2 * j] = wd.mean;
      buf[2 * j + 1] = wd.var;
      buf[4 + j] = wd.count;
    }
    __syncthreads();
    if (lane == 0 && warp < offset) {
      Stats other{buf[2 * warp], buf[2 * warp + 1], buf[4 + warp]};
      wd = combine(wd, other);
    }
    __syncthreads();
  }
  if (lane == 0 && warp == 0) {
    buf[0] = wd.mean;
    buf[1] = wd.var / 1024.f;
  }
  __syncthreads();
  return {buf[0], buf[1], 0};
}

struct HalfLoad {
  const half* p;
  __device__ float operator()(int j) const {
    return __half2float(p[j]);
  }
};

struct FloatLoad {
  const float* p;
  __device__ float operator()(int j) const {
    return p[j];
  }
};

__global__ void embed_norm(const int64_t* ids, const half* w, const float* gamma,
                           float* x, BF* y) {
  int r = blockIdx.x, t = threadIdx.x + threadIdx.y * 32;
  __shared__ float buf[6];
  HalfLoad load{w + ids[r] * 1024};
  Stats wd = stats(load, buf);
  float inv = rsqrtf(wd.var + 1e-5f);
  for (int i = t; i < 256; i += 128) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      int j = 4 * i + k;
      float z = gamma[j] * (inv * (load(j) - wd.mean));
      x[r * 1024 + j] = z;
      y[r * 1024 + j] = __float2bfloat16_rn(z);
    }
  }
}

__global__ void add_type(const BF* y, const BF* emb, const int64_t* types,
                         float* x, int L, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total)
    x[i] = __bfloat162float(y[i]) +
           __bfloat162float(emb[types[i / (L * 1024)] * 1024 + i % 1024]);
}

__global__ void add_residual(float* x, const BF* y, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total)
    x[i] += __bfloat162float(y[i]);
}

__global__ void gather_norm(const float* h, const int32_t* indices,
                            const float* gamma, const float* bias, BF* y) {
  int r = blockIdx.x, t = threadIdx.x + threadIdx.y * 32;
  __shared__ float buf[6];
  FloatLoad load{h + indices[r] * 1024};
  Stats wd = stats(load, buf);
  float inv = rsqrtf(wd.var + 1e-5f);
  for (int i = t; i < 256; i += 128) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      int j = 4 * i + k;
      y[r * 1024 + j] =
          __float2bfloat16_rn(gamma[j] * (inv * (load(j) - wd.mean)) + bias[j]);
    }
  }
}

// Match torch.addmm: accumulate and add bias in FP32, then round once to BF16.
__global__ void linear_finish(const float* acc, const BF* bias, BF* out, int n,
                              int count, int activation) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    BF rounded = __float2bfloat16_rn(acc[i] + __bfloat162float(bias[i % n]));
    if (activation) {
      float v = __bfloat162float(rounded);
      rounded = __float2bfloat16_rn(
          0.5f * v * (1.0f + erff(v * 0.7071067811865476f)));
    }
    out[i] = rounded;
  }
}

struct LinearContext {
  cublasHandle_t handle;
  float* scratch;
};

__global__ void action_features(const float* h, const BF* logits,
                                const int32_t* offsets, BF* out, int L) {
  int b = blockIdx.x, t = threadIdx.x;
  int lo = offsets[b], hi = offsets[b + 1];
  for (int j = t; j < 1024; j += blockDim.x)
    out[b * 1028 + j] = __float2bfloat16_rn(h[b * L * 1024 + j]);
  if (t == 0) {
    float maxv = -INFINITY;
    for (int i = lo; i < hi; i++)
      maxv = fmaxf(maxv, __bfloat162float(logits[i]));
    float sum = 0;
    for (int i = lo; i < hi; i++)
      sum += expf(__bfloat162float(logits[i]) - maxv);
    float top1 = 0, top2 = 0, entropy = 0;
    for (int i = lo; i < hi; i++) {
      float p = expf(__bfloat162float(logits[i]) - maxv) / sum;
      entropy -= p * logf(fmaxf(p, 1e-9f));
      if (p > top1) {
        top2 = top1;
        top1 = p;
      } else if (p > top2)
        top2 = p;
    }
    int k = max(2, hi - lo);
    out[b * 1028 + 1024] = __float2bfloat16_rn(top1);
    out[b * 1028 + 1025] = __float2bfloat16_rn(top1 - top2);
    out[b * 1028 + 1026] = __float2bfloat16_rn(entropy / logf(float(k)));
    out[b * 1028 + 1027] = __float2bfloat16_rn(float(k) / 255.0f);
  }
}

extern "C" {
int laya_embed(void** p, int B, int L, int M, cudaStream_t s) {
  embed_norm<<<M, dim3(32, 4), 0, s>>>(
      (int64_t*)p[0], (half*)p[1], (float*)p[2], (float*)p[3], (BF*)p[4]);
  return cudaGetLastError();
}

int laya_type(void** p, int B, int L, int M, cudaStream_t s) {
  add_type<<<(M * 1024 + 255) / 256, 256, 0, s>>>(
      (BF*)p[0], (BF*)p[1], (int64_t*)p[2], (float*)p[3], L, M * 1024);
  return cudaGetLastError();
}

int laya_residual(void** p, int B, int L, int M, cudaStream_t s) {
  add_residual<<<(M * 1024 + 255) / 256, 256, 0, s>>>(
      (float*)p[0], (BF*)p[1], M * 1024);
  return cudaGetLastError();
}

int laya_gather(void** p, int B, int L, int M, cudaStream_t s) {
  gather_norm<<<M, dim3(32, 4), 0, s>>>(
      (float*)p[0], (int32_t*)p[1], (float*)p[2], (float*)p[3], (BF*)p[4]);
  return cudaGetLastError();
}

int laya_features(void** p, int B, int L, int M, cudaStream_t s) {
  action_features<<<B, 256, 0, s>>>(
      (float*)p[0], (BF*)p[1], (int32_t*)p[2], (BF*)p[3], L);
  return cudaGetLastError();
}

int laya_blas_create(void** out, cudaStream_t s) {
  *out = nullptr;
  auto* h = new LinearContext{};
  auto rc = cublasCreate(&h->handle);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    delete h;
    return 20000 + rc;
  }
  rc = cublasSetStream(h->handle, s);
  if (rc != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(h->handle);
    delete h;
    return 20000 + rc;
  }
  auto ce = cudaMalloc(&h->scratch, 2048 * 1024 * sizeof(float));
  if (ce != cudaSuccess) {
    cublasDestroy(h->handle);
    delete h;
    return ce;
  }
  *out = h;
  return 0;
}

int laya_blas_free(void* ptr) {
  auto* h = (LinearContext*)ptr;
  auto ce = cudaFree(h->scratch);
  auto rc = cublasDestroy(h->handle);
  delete h;
  return ce != cudaSuccess ? int(ce) : (rc != CUBLAS_STATUS_SUCCESS ? 20000 + rc : 0);
}

int laya_linear(void* ptr, const void* a, const void* w, const void* bias,
                void* out, int rows, int n, int k, int activation, cudaStream_t s) {
  if (rows < 1 || rows > 2048 || n < 1 || n > 1024 || k < 1)
    return -1;
  auto* h = (LinearContext*)ptr;
  float alpha = 1, beta = 0;
  auto rc = cublasGemmEx(
      h->handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k, &alpha, w, CUDA_R_16BF, k,
      a, CUDA_R_16BF, k, &beta, h->scratch, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (rc != CUBLAS_STATUS_SUCCESS)
    return 20000 + rc;
  linear_finish<<<(rows * n + 255) / 256, 256, 0, s>>>(
      h->scratch, (const BF*)bias, (BF*)out, n, rows * n, activation);
  return cudaGetLastError();
}
}
