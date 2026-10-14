// The unfused baseline the fused candidate-scoring kernel is measured against.
//
// Cosine similarity needs each candidate's norm and its dot with the query, and
// both read the same elements of `candidates`. The fused kernel accumulates them
// in one read of the row. This baseline is the same computation split the way it
// falls out of composing existing pieces: a pass that normalises the candidate
// matrix, then a pass that scores it. That reads `candidates` twice.
//
// Everything else is deliberately identical to `candidate_scoring.cu` -- the same
// block shape, the same warp reduction, the same thread-0 softmax -- so the only
// difference the benchmark can see is the second pass over `candidates`.
//
// It is the *conservative* baseline. A real unfused pipeline built from a GEMM
// would also round-trip the similarity matrix through global memory, which this
// does not, so the difference measured here is a floor.
//
// Not part of libscoring.so: `build.sh` compiles only `candidate_scoring.cu` and
// the manifest lists only that file. `bench.py` compiles this one itself.
#include <cuda_runtime.h>

namespace {

constexpr int WARP = 32;
constexpr int THREADS = 256;
constexpr int MAX_K = 255;
constexpr float NORM_EPS = 1e-12f;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Pass one: the norm of every candidate row, one block per row. This is the read
// the fusion removes.
__global__ void row_norms_kernel(const float* __restrict__ candidates,
                                 float* __restrict__ norms, int rows, int D) {
    __shared__ float partials[THREADS / WARP];
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const float* row = candidates + static_cast<size_t>(blockIdx.x) * D;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < D; d += THREADS) {
        const float value = row[d];
        sum = fmaf(value, value, sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        partials[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float total = (lane < blockDim.x / WARP) ? partials[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) {
            norms[blockIdx.x] = fmaxf(sqrtf(total), NORM_EPS);
        }
    }
}

// Pass two: the fused kernel's block, with the row norm loaded instead of
// accumulated alongside the dot.
__global__ void scoring_unfused_kernel(const float* __restrict__ query,
                                       const float* __restrict__ candidates,
                                       const float* __restrict__ row_norms,
                                       float* __restrict__ probabilities, int K, int D,
                                       float scale, float temperature, int normalize) {
    __shared__ float similarity[MAX_K];
    __shared__ float reduce_buffer[THREADS / WARP];

    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int warps = blockDim.x / WARP;

    const float* question_query = query + static_cast<size_t>(blockIdx.x) * D;
    const float* question_candidates = candidates + static_cast<size_t>(blockIdx.x) * K * D;
    const float* question_norms = row_norms + static_cast<size_t>(blockIdx.x) * K;
    float* question_probabilities = probabilities + static_cast<size_t>(blockIdx.x) * K;

    float query_norm = 1.0f;
    if (normalize) {
        if (warp == 0) {
            float partial = 0.0f;
            for (int d = lane; d < D; d += WARP) {
                const float q = question_query[d];
                partial = fmaf(q, q, partial);
            }
            partial = warp_sum(partial);
            if (lane == 0) {
                reduce_buffer[0] = fmaxf(sqrtf(partial), NORM_EPS);
            }
        }
        __syncthreads();
        query_norm = reduce_buffer[0];
        __syncthreads();
    }

    for (int k = warp; k < K; k += warps) {
        const float* row = question_candidates + static_cast<size_t>(k) * D;
        float dot = 0.0f;
        for (int d = lane; d < D; d += WARP) {
            dot = fmaf(row[d], question_query[d], dot);
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            float value = dot;
            if (normalize) {
                value /= query_norm * question_norms[k];
            }
            similarity[k] = value * scale / temperature;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float maximum = similarity[0];
        for (int k = 1; k < K; ++k) {
            maximum = fmaxf(maximum, similarity[k]);
        }
        float total = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float value = expf(similarity[k] - maximum);
            similarity[k] = value;
            total += value;
        }
        const float inverse = (total > 0.0f) ? (1.0f / total) : 0.0f;
        for (int k = 0; k < K; ++k) {
            question_probabilities[k] = similarity[k] * inverse;
        }
    }
}

}  // namespace

extern "C" {

// Two passes over `candidates`, with the row norms staged in `norms`.
int bench_unfused(const float* query, const float* candidates, float* norms,
                  float* probabilities, int questions, int K, int D, float scale,
                  float temperature, int normalize, cudaStream_t stream) {
    // The norm pass exists only because a cosine needs one. A plain dot does not,
    // and running it anyway would make the baseline slower for a reason the fused
    // kernel never removed.
    if (normalize) {
        row_norms_kernel<<<questions * K, THREADS, 0, stream>>>(candidates, norms,
                                                                questions * K, D);
    }
    scoring_unfused_kernel<<<questions, THREADS, 0, stream>>>(query, candidates, norms,
                                                              probabilities, K, D, scale,
                                                              temperature, normalize);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
