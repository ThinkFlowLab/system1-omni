// Fused candidate scoring: query + candidate matrix -> probabilities, one pass.
//
// What this replaces is several launches over an intermediate the caller never
// wanted: materialise similarities, find the max, exponentiate, sum, divide. For
// a decision model that is most of the readout, and the candidate matrix is read
// once per stage instead of once in total.
//
// The fusion that matters here is narrower and more useful than "fewer
// launches": when the similarity is a cosine, each candidate's norm and its dot
// with the query need exactly the same elements, so both accumulate in one read
// of C. That halves the traffic over the candidate matrix, which for Kev means
// 255 rows of 2560 floats.
//
// Layout: one thread block per question; each warp owns candidates in a strided
// loop and reduces its dot with warp shuffles; the block then does a stable
// softmax over the K similarities in shared memory. K is at most 255, so the
// whole similarity vector fits in shared memory and the softmax needs no second
// kernel and no atomics. Batching is a grid over questions, not a loop of
// launches.
//
// Accumulation is float32, which is what a model engine would do; the reference
// in reference.py accumulates in float64 so the error is measured against the
// true value rather than against another float32 path.
//
// NOT COMPILED OR RUN. There is no CUDA toolkit on the authoring machine. See
// README.md for exactly what is and is not verified.

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int WARP = 32;
constexpr int THREADS = 1024;
constexpr int MAX_K = 255;      // the largest candidate set the serving API allows
constexpr float NORM_EPS = 1e-12f;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One block per question, so a batch is a grid rather than a loop of launches.
// Every warp walks candidates k = warp, warp + warps, ...
__global__ void candidate_scoring_kernel(const float* __restrict__ query,
                                         const float* __restrict__ candidates,
                                         float* __restrict__ probabilities,
                                         float* __restrict__ logits_out,
                                         int K, int D, float scale, float temperature,
                                         int normalize) {
    __shared__ float similarity[MAX_K];
    __shared__ float reduce_buffer[THREADS / WARP];

    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int warps = blockDim.x / WARP;

    const float* question_query = query + static_cast<size_t>(blockIdx.x) * D;
    const float* question_candidates =
        candidates + static_cast<size_t>(blockIdx.x) * K * D;
    float* question_probabilities =
        probabilities + static_cast<size_t>(blockIdx.x) * K;
    float* question_logits =
        (logits_out == nullptr) ? nullptr
                                : logits_out + static_cast<size_t>(blockIdx.x) * K;

    // The query norm is shared by every candidate, so one warp computes it and
    // the block reads it.
    //
    // It must be ONE warp, not all of them. The inner loop walks
    // `d = lane; d < D; d += WARP`, which already covers every element of D
    // across the 32 lanes of a single warp. Having each warp compute the same
    // partial and then summing the partials across warps counts the sum of
    // squares `warps` times, so the norm came out sqrt(8) too large on a
    // 256-thread block -- which is what the first GPU run measured: the
    // normalize cases were off by ~1e-2 while every unnormalized case was at
    // float32 rounding.
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
                // Floor the norm, not the squared norm, so this matches the
                // reference exactly: it computes sqrt(sum) and then takes the
                // maximum with eps. Flooring the square instead would differ on
                // the zero-candidate case, which is in the fixed vectors.
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
        float row_norm2 = 0.0f;
        // One read of the row yields both the dot and the norm.
        for (int d = lane; d < D; d += WARP) {
            const float c = row[d];
            dot = fmaf(c, question_query[d], dot);
            if (normalize) {
                row_norm2 = fmaf(c, c, row_norm2);
            }
        }
        dot = warp_sum(dot);
        if (normalize) {
            row_norm2 = warp_sum(row_norm2);
        }
        if (lane == 0) {
            float value = dot;
            if (normalize) {
                value /= query_norm * fmaxf(sqrtf(row_norm2), NORM_EPS);
            }
            similarity[k] = value * scale / temperature;
        }
    }
    __syncthreads();

    // Stable softmax over the K values already in shared memory: max, then the
    // sum of exponentials, then divide. Subtracting the max is what keeps a
    // large scale from overflowing to inf; the reference fixes that as the
    // convention, and reference.py's overflow-prone case is the check for it.
    if (threadIdx.x == 0) {
        float maximum = similarity[0];
        for (int k = 1; k < K; ++k) {
            maximum = fmaxf(maximum, similarity[k]);
        }
        float total = 0.0f;
        for (int k = 0; k < K; ++k) {
            // The logit is recorded before the slot is overwritten with its
            // exponential, so the optional output really is the pre-softmax
            // value.
            if (question_logits != nullptr) {
                question_logits[k] = similarity[k];
            }
            const float value = expf(similarity[k] - maximum);
            similarity[k] = value;
            total += value;
        }
        // K is at least 1 and every exponential is at least exp(-max_spread),
        // so total is positive for finite input; the guard keeps a NaN input
        // from turning into a division by zero on top of it.
        const float inverse = (total > 0.0f) ? (1.0f / total) : 0.0f;
        for (int k = 0; k < K; ++k) {
            question_probabilities[k] = similarity[k] * inverse;
        }
    }
}

bool arguments_valid(int K, int D, float scale, float temperature) {
    // MAX_K is the API's limit, not this kernel's convenience: silently scoring
    // the first 255 of a larger set would return a confident wrong answer
    // instead of an error. A non-positive scale or temperature is a caller bug
    // with no sensible default.
    return K > 0 && K <= MAX_K && D > 0 && scale > 0.0f && temperature > 0.0f;
}

}  // namespace

extern "C" {

// ABI version of this library; a loader refuses a value it does not know.
uint32_t cs_score_abi_version(void) { return 1; }

// Scores `questions` independent questions.
//
//   query         [questions, D]              row-major float32
//   candidates    [questions, K, D]           row-major float32, K <= 255
//   probabilities [questions, K]              written; each row sums to 1
//   logits_out    [questions, K] or null      pre-softmax values
//
// Returns 0, or a cudaError_t. Work is queued on `stream` and the call does not
// synchronize, so a caller can capture it into a CUDA Graph. A single question
// is `questions = 1`.
int cs_score_candidates_batch(const float* query, const float* candidates,
                              float* probabilities, float* logits_out, int questions,
                              int K, int D, float scale, float temperature, int normalize,
                              cudaStream_t stream) {
    if (query == nullptr || candidates == nullptr || probabilities == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (questions <= 0 || !arguments_valid(K, D, scale, temperature)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    candidate_scoring_kernel<<<questions, THREADS, 0, stream>>>(
        query, candidates, probabilities, logits_out, K, D, scale, temperature,
        normalize ? 1 : 0);
    return static_cast<int>(cudaGetLastError());
}

// Convenience for the common single-question call.
int cs_score_candidates(const float* query, const float* candidates, float* probabilities,
                        float* logits_out, int K, int D, float scale, float temperature,
                        int normalize, cudaStream_t stream) {
    return cs_score_candidates_batch(query, candidates, probabilities, logits_out, 1, K, D,
                                     scale, temperature, normalize, stream);
}

}  // extern "C"
