// Test-only loader wrapper. All other cs1 symbols come from the linked real library.
// Fail instantiation after real capture, destroy the executable, and leave math intact.
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

extern "C" int cs1_graph_end(void* stream, void** exec) {
    const char* path = std::getenv("CUA_S1_REAL_CUDA_LIB");
    if (!path) std::abort();
    void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) std::abort();
    using End = int (*)(void*, void**);
    using Destroy = int (*)(void*);
    auto end = reinterpret_cast<End>(dlsym(library, "cs1_graph_end"));
    auto destroy = reinterpret_cast<Destroy>(dlsym(library, "cs1_graph_destroy"));
    if (!end || !destroy || end == &cs1_graph_end) std::abort();
    int result = end(stream, exec);
    if (result == 0) {
        if (*exec && destroy(*exec) != 0) std::abort();
        *exec = nullptr;
        std::fputs("Injected vision Graph instantiation failure\n", stderr);
        result = 2; // cudaErrorMemoryAllocation; no CUDA operation was changed.
    }
    dlclose(library);
    return result;
}
