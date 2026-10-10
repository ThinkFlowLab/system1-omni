// Test-only wrapper: real CUDA math, fail the second selected-head projection.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include "ops.h"

typedef int (*gemm_fn)(void*, const void*, const void*, void*, int, int, int, int, void*);
int cs1_gemm(void* handle, const void* x, const void* w, void* y,
             int m, int n, int k, int ldy, void* stream) {
    static unsigned projection_calls;
    gemm_fn original = (gemm_fn)dlsym(RTLD_NEXT, "cs1_gemm");
    if (!original) return 999;
    if (m >= 1 && m <= 16 && n == 256 && k == 2048) {
        fprintf(stderr, "Decider test projection %u rows %d\n", ++projection_calls, m);
        if (projection_calls == 2) return 2;
    }
    return original(handle, x, w, y, m, n, k, ldy, stream);
}
