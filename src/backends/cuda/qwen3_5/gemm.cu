// bfloat16 GEMMs through cuBLASLt, float32 accumulation.
//
// Row-major y [M, N] = x [M, K] * w [N, K]^T is the column-major product
// y^T [N, M] = (w viewed as [K, N])^T * (x viewed as [K, M]); y's rows may be
// strided (ldy >= N), so one GEMM can fill a slice of a wider buffer.
//
// Each shape uses cuBLASLt's first heuristic choice, excluding split-K reductions that
// accumulate into the output in place, since their order, and so the rounding, is not
// fixed (vision's FP32 and biased GEMMs exclude all split-K). That choice depends on M,
// so a row's result can change with the number of rows in the call. A handle from
// cs1_gemm_create_fixed instead keeps one algorithm per weight shape (N, K, ldy, and for
// vision's GEMMs the data type and bias) for every M: the heuristic's first choice at a
// reference M among algorithms without split-K. Each output row then takes the same
// path whatever the other rows, so its result does not depend on M or on its row index.
// cuBLASLt does not document this; tests/qwen3_5/kernels.rs checks it on the GPU it
// runs on.
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

using Key = std::tuple<int, int, int, int, bool, bool>;  // M, N, K, ldy, FP32, bias

struct Gemm {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    int reference_m = 0;  // > 0: one algorithm per weight shape, chosen at this M
    std::map<Key, Plan> plans;
    std::map<std::tuple<int, int, int, bool, bool>, cublasLtMatmulAlgo_t> fixed;  // N, K, ldy, FP32, bias
};

int status(cublasStatus_t s) { return s == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + (int)s; }

void destroy(Plan& p) {
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    p = Plan{};
}

int describe(int M, int N, int K, int ldy, Plan& p, bool fp32, bool bias) {
    cublasStatus_t s = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    if (bias) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        if ((s = cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))) != CUBLAS_STATUS_SUCCESS) return status(s);
    }
    const cudaDataType_t dtype = fp32 ? CUDA_R_32F : CUDA_R_16BF;
    if ((s = cublasLtMatrixLayoutCreate(&p.a, dtype, K, N, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.b, dtype, K, M, K)) != CUBLAS_STATUS_SUCCESS) return status(s);
    if ((s = cublasLtMatrixLayoutCreate(&p.c, dtype, N, M, ldy)) != CUBLAS_STATUS_SUCCESS) return status(s);
    return 0;
}

// The heuristic's first choice among the given reduction schemes.
int first_choice(Gemm& g, Plan& p, uint32_t schemes) {
    cublasLtMatmulPreference_t pref;
    cublasStatus_t s = cublasLtMatmulPreferenceCreate(&pref);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.workspace_bytes,
                                         sizeof(g.workspace_bytes));
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

// The weight shape's algorithm, chosen at the reference M on first use. An M it cannot
// serve is an error, not a reason to switch algorithms.
int fixed_choice(Gemm& g, int N, int K, int ldy, bool fp32, bool bias, Plan& p) {
    const std::tuple<int, int, int, bool, bool> key{N, K, ldy, fp32, bias};
    auto it = g.fixed.find(key);
    if (it == g.fixed.end()) {
        Plan r;
        int rc = describe(g.reference_m, N, K, ldy, r, fp32, bias);
        if (rc == 0) rc = first_choice(g, r, CUBLASLT_REDUCTION_SCHEME_NONE);
        if (rc == 0) {
            // Check that the choice really has no split-K reduction.
            int32_t splits = 0;
            uint32_t scheme = 0;
            size_t written = 0;
            const bool read =
                cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits,
                                                     sizeof(splits), &written) == CUBLAS_STATUS_SUCCESS &&
                cublasLtMatmulAlgoConfigGetAttribute(&r.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &scheme,
                                                     sizeof(scheme), &written) == CUBLAS_STATUS_SUCCESS;
            if (!read || splits != 1 || scheme != CUBLASLT_REDUCTION_SCHEME_NONE)
                rc = status(CUBLAS_STATUS_NOT_SUPPORTED);
        }
        if (rc == 0) it = g.fixed.emplace(key, r.algo).first;
        destroy(r);
        if (rc != 0) return rc;
    }
    p.algo = it->second;
    cublasLtMatmulHeuristicResult_t check{};
    const cublasStatus_t s = cublasLtMatmulAlgoCheck(g.handle, p.op, p.a, p.b, p.c, p.c, &p.algo, &check);
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    if (check.workspaceSize > g.workspace_bytes) return status(CUBLAS_STATUS_NOT_SUPPORTED);
    return 0;
}

// The plan for a shape, created on first use.
int plan_for(Gemm& g, int M, int N, int K, int ldy, Plan*& out, bool fp32 = false, bool bias = false) {
    const Key key{M, N, K, ldy, fp32, bias};
    auto it = g.plans.find(key);
    if (it != g.plans.end()) {
        out = &it->second;
        return 0;
    }
    Plan p;
    int rc = describe(M, N, K, ldy, p, fp32, bias);
    if (rc == 0 && g.reference_m > 0) {
        rc = fixed_choice(g, N, K, ldy, fp32, bias, p);
    } else if (rc == 0) {
        // Vision's FP32 and biased GEMMs exclude all split-K to avoid BF16 intermediate
        // reductions; the language GEMMs exclude only in-place reductions.
        rc = first_choice(g, p, fp32 || bias ? CUBLASLT_REDUCTION_SCHEME_NONE
                                             : CUBLASLT_REDUCTION_SCHEME_MASK & ~CUBLASLT_REDUCTION_SCHEME_INPLACE);
    }
    if (rc != 0) {
        destroy(p);
        return rc;
    }
    out = &g.plans.emplace(key, p).first->second;
    return 0;
}

}  // namespace

extern "C" void* cs1_gemm_create_fixed(size_t workspace_bytes, int reference_m) {
    if (reference_m <= 0) return nullptr;
    Gemm* g = static_cast<Gemm*>(cs1_gemm_create(workspace_bytes));
    if (g) g->reference_m = reference_m;
    return g;
}

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

// Vision's biased BF16 linears round only after adding bias. The patch
// convolution passes zero bias here and applies its bias after BF16 rounding.
extern "C" int cs1_vision_linear(void* gemm, const void* x, const void* w, const void* bias,
                                  void* y, int M, int N, int K, void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || !bias || M <= 0 || N <= 0 || K <= 0) return cudaErrorInvalidValue;
    Plan* p = nullptr;
    int rc = plan_for(*g, M, N, K, N, p, false, true);
    if (rc) return rc;
    auto s = cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
    if (s != CUBLAS_STATUS_SUCCESS) return status(s);
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c, &p->algo,
                                 g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}
// Separate unmerged LoRA matrices and intermediates stay FP32. No TF32 fast compute.
extern "C" int cs1_gemm_f32(void* gemm, const float* x, const float* w, float* y,
                             int M, int N, int K, void* stream) {
    Gemm* g = static_cast<Gemm*>(gemm);
    if (!g || M <= 0 || N <= 0 || K <= 0) return cudaErrorInvalidValue;
    Plan* p = nullptr;
    int rc = plan_for(*g, M, N, K, N, p, true, false);
    if (rc) return rc;
    const float alpha = 1.f, beta = 0.f;
    return status(cublasLtMatmul(g->handle, p->op, &alpha, w, p->a, x, p->b, &beta, y, p->c, y, p->c, &p->algo,
                                 g->workspace, g->workspace_bytes, static_cast<cudaStream_t>(stream)));
}
