// bfloat16 GEMMs through cuBLASLt, float32 accumulation.
//
// Row-major y [M, N] = x [M, K] * w [N, K]^T is the column-major product
// y^T [N, M] = (w viewed as [K, N])^T * (x viewed as [K, M]); y's rows may be
// strided (ldy >= N), so one GEMM can fill a slice of a wider buffer.
//
// cs1_gemm_tune times cuBLASLt's shortlist for a shape (L2 flushed before each call)
// and keeps the fastest if it beats the heuristic's first choice by more than 3%. A
// shape that was not tuned borrows the choice for a nearby tuned M with the same N, K
// and ldy, or takes the first choice. Split-K reductions that accumulate into the
// output in place are excluded, since their order, and so the rounding, is not fixed.
#include <cublasLt.h>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

#include "ops.h"

namespace {

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool tuned = false;
};

using Key = std::tuple<int, int, int, int>;  // M, N, K, ldy

struct Gemm {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    std::map<Key, Plan> plans;
    // tuning only: a buffer larger than L2, a sink for its reads, and two events
    void* flush = nullptr;
    int* sink = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
};

void release_tuning(Gemm& g) {
    if (g.flush) cudaFree(g.flush);
    if (g.sink) cudaFree(g.sink);
    if (g.e0) cudaEventDestroy(g.e0);
    if (g.e1) cudaEventDestroy(g.e1);
    g.flush = nullptr;
    g.sink = nullptr;
    g.e0 = g.e1 = nullptr;
}

int status(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + (int)s; }

void destroy(Plan& p) {
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    p = Plan{};
}

int describe(int M, int N, int K, int ldy, Plan& p) {
    cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    if ((s = cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, ldy)) != CUBLAS_STATUS_SUCCESS) return status(s);
    return 0;
}

// Up to `want` heuristic choices, best first, without in-place split-K reductions.
int heuristics(Gemm& g, const Plan& p, int want, std::vector<cublasLtMatmulAlgo_t>& out) {
    cublasLtMatmulPreference_t pref;
    cublasStatus_t s = cublasLtMatmulPreferenceCreate(&pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.workspace_bytes,
                                         sizeof(g.workspace_bytes));
    const uint32_t schemes = CUBLASLT_REDUCTION_SCHEME_MASK & ~CUBLASLT_REDUCTION_SCHEME_INPLACE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &schemes,
                                         sizeof(schemes));
    std::vector<cublasLtMatmulHeuristicResult_t> r(want);
    int found = 0;
    s = cublasLtMatmulAlgoGetHeuristic(g.handle, p.op, p.a, p.b, p.c, p.c, pref, want, r.data(), &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    for (int i = 0; i < found; i++)
        if (r[i].state == CUBLAS_STATUS_SUCCESS) out.push_back(r[i].algo);
    return out.empty() ? status(CUBLAS_STATUS_NOT_SUPPORTED) : 0;
}

bool usable(Gemm& g, const Plan& p, const cublasLtMatmulAlgo_t& algo) {
    cublasLtMatmulHeuristicResult_t r{};
    return cublasLtMatmulAlgoCheck(g.handle, p.op, p.a, p.b, p.c, p.c, &algo, &r) == CUBLAS_STATUS_SUCCESS &&
           r.workspaceSize <= g.workspace_bytes;
}

// Read a buffer larger than L2, so the next call finds none of its operands cached.
__global__ void flush_l2(const int4* p, size_t n, int* sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc ^= p[i].x ^ p[i].w;
    if (acc == 0x7fffffff) *sink = acc;
}

constexpr size_t FLUSH_BYTES = 256u << 20;

// The plan for a shape, created on first use: the choice for the smallest tuned M above
// if that is at most twice this M, else for the largest tuned M below if none is above,
// else the heuristic's first choice.
int plan_for(Gemm& g, int M, int N, int K, int ldy, Plan*& out) {
    const Key key{M, N, K, ldy};
    auto it = g.plans.find(key);
    if (it != g.plans.end()) {
        out = &it->second;
        return 0;
    }
    Plan p;
    int rc = describe(M, N, K, ldy, p);
    const Plan *above = nullptr, *below = nullptr;
    int above_m = 0, below_m = 0;
    for (auto& [k, q] : g.plans) {
        const auto [m, n, kk, l] = k;
        if (!q.tuned || n != N || kk != K || l != ldy) continue;
        if (m > M && (!above || m < above_m)) above = &q, above_m = m;
        if (m < M && (!below || m > below_m)) below = &q, below_m = m;
    }
    const Plan* borrow = above && above_m <= 2 * M ? above : above ? nullptr : below;
    if (rc == 0 && borrow && usable(g, p, borrow->algo)) {
        p.algo = borrow->algo;
    } else if (rc == 0) {
        std::vector<cublasLtMatmulAlgo_t> first;
        rc = heuristics(g, p, 1, first);
        if (rc == 0) p.algo = first[0];
    }
    if (rc != 0) {
        destroy(p);
        return rc;
    }
    out = &g.plans.emplace(key, p).first->second;
    return 0;
}

}  // namespace

extern "C" void* cs1_gemm_create(size_t workspace_bytes) {
    Gemm* g = new Gemm();
    if (cublasLtCreate(&g->handle) != CUBLAS_STATUS_SUCCESS ||
        (workspace_bytes > 0 && cudaMalloc(&g->workspace, workspace_bytes) != cudaSuccess)) {
        if (g->handle) cublasLtDestroy(g->handle);
        delete g;
        return nullptr;
    }
    g->workspace_bytes = workspace_bytes;
    return g;
}

extern "C" void cs1_gemm_destroy(void* gemm) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g) return;
    for (auto& kv : g->plans) destroy(kv.second);
    release_tuning(*g);
    if (g->workspace) cudaFree(g->workspace);
    cublasLtDestroy(g->handle);
    delete g;
}

extern "C" int cs1_gemm_tune(void* gemm, const void* x, const void* w, void* y, int M, int N, int K, int ldy,
                             void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || M <= 0 || N <= 0 || K <= 0 || ldy < N) return cudaErrorInvalidValue;
    const Key key{M, N, K, ldy};
    if (auto it = g->plans.find(key); it != g->plans.end()) {
        destroy(it->second);
        g->plans.erase(it);
    }
    Plan p;
    std::vector<cublasLtMatmulAlgo_t> cands;
    int rc = describe(M, N, K, ldy, p);
    if (rc == 0) rc = heuristics(*g, p, 16, cands);
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    if (rc == 0 && !g->flush &&
        (cudaMalloc(&g->flush, FLUSH_BYTES) || cudaMalloc(&g->sink, sizeof(int)) ||
         cudaMemsetAsync(g->flush, 0, FLUSH_BYTES, st) ||
         cudaEventCreate(&g->e0) || cudaEventCreate(&g->e1))) {
        release_tuning(*g);
        cudaGetLastError();
        rc = cudaErrorMemoryAllocation;
    }
    if (rc != 0) {
        destroy(p);
        return rc;
    }
    const float alpha = 1.f, beta = 0.f;
    // the median of `reps` calls, each after an L2 flush; huge if the call fails
    auto time = [&](const cublasLtMatmulAlgo_t& algo, int reps) {
        std::vector<float> times;
        for (int r = 0; r < reps; r++) {
            flush_l2<<<1024, 256, 0, st>>>(static_cast<const int4*>(g->flush), FLUSH_BYTES / sizeof(int4), g->sink);
            cudaEventRecord(g->e0, st);
            const cublasStatus_t s = cublasLtMatmul(g->handle, p.op, &alpha, w, p.a, x, p.b, &beta, y, p.c, y, p.c,
                                                    &algo, g->workspace, g->workspace_bytes, st);
            cudaEventRecord(g->e1, st);
            float ms = 0.f;
            if (s != CUBLAS_STATUS_SUCCESS || cudaEventSynchronize(g->e1) != cudaSuccess) {
                cudaGetLastError();
                return 1e30f;
            }
            cudaEventElapsedTime(&ms, g->e0, g->e1);
            times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        return times[reps / 2];
    };
    // nine timed calls each, or three for shapes that take over 2 ms
    const int reps = time(cands[0], 1) > 2.f ? 3 : 9;
    std::vector<float> median;
    for (auto& a : cands) median.push_back(time(a, reps));
    const size_t fastest = std::min_element(median.begin(), median.end()) - median.begin();
    if (median[fastest] >= 1e30f) {
        destroy(p);
        return status(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    p.algo = cands[median[fastest] < 0.97f * median[0] ? fastest : 0];
    p.tuned = true;
    g->plans.emplace(key, p);
    return (int)cudaGetLastError();
}

extern "C" void cs1_gemm_tune_done(void* gemm) {
    if (gemm) release_tuning(*static_cast<Gemm*>(gemm));
}

extern "C" int cs1_gemm(void* gemm, const void* x, const void* w, void* y, int M, int N, int K, int ldy,
                        void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || M < 0 || N <= 0 || K <= 0 || ldy < N) return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    Plan* p = nullptr;
    const int rc = plan_for(*g, M, N, K, ldy, p);
    if (rc != 0) return rc;
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c, &p->algo,
                                 g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}
