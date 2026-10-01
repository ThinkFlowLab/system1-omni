// bfloat16 GEMMs through cuBLASLt, float32 accumulation.
//
// Row-major y [M, N] = x [M, K] * w [N, K]^T is the column-major product
// y^T [N, M] = (w viewed as [K, N])^T * (x viewed as [K, M]); y's rows may be
// strided (ldy >= N), so one GEMM can fill a slice of a wider buffer.
//
// Each shape uses cuBLASLt's first heuristic choice, excluding split-K reductions that
// accumulate into the output in place, since their order, and so the rounding, is not
// fixed.
#include <cublasLt.h>

#include <map>
#include <tuple>

#include "ops.h"

namespace {

struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

using Key = std::tuple<int, int, int, int>;  // M, N, K, ldy

struct Gemm {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    std::map<Key, Plan> plans;
};

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

// The heuristic's first choice, without in-place split-K reductions.
int first_choice(Gemm& g, Plan& p) {
    cublasLtMatmulPreference_t pref;
    cublasStatus_t s = cublasLtMatmulPreferenceCreate(&pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.workspace_bytes,
                                         sizeof(g.workspace_bytes));
    const uint32_t schemes = CUBLASLT_REDUCTION_SCHEME_MASK & ~CUBLASLT_REDUCTION_SCHEME_INPLACE;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &schemes,
                                         sizeof(schemes));
    cublasLtMatmulHeuristicResult_t r{};
    int found = 0;
    s = cublasLtMatmulAlgoGetHeuristic(g.handle, p.op, p.a, p.b, p.c, p.c, pref, 1, &r, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    if (found == 0 || r.state != CUBLAS_STATUS_SUCCESS) return status(CUBLAS_STATUS_NOT_SUPPORTED);
    p.algo = r.algo;
    return 0;
}

// The plan for a shape, created on first use.
int plan_for(Gemm& g, int M, int N, int K, int ldy, Plan*& out) {
    const Key key{M, N, K, ldy};
    auto it = g.plans.find(key);
    if (it != g.plans.end()) {
        out = &it->second;
        return 0;
    }
    Plan p;
    int rc = describe(M, N, K, ldy, p);
    if (rc == 0) rc = first_choice(g, p);
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
    if (g->workspace) cudaFree(g->workspace);
    cublasLtDestroy(g->handle);
    delete g;
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
