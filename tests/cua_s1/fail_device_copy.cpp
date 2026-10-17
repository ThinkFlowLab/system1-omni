// Test-only wrapper: enqueue the real D2D copy, fail once, and record its drain.
// Every CUDA operation delegates to the ABI-matched real library.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <stddef.h>
namespace {
std::atomic<bool> injected{false};
std::atomic<bool> pending{false};
void* real_library() {
    const char* path = std::getenv("CUA_S1_REAL_CUDA_LIB");
    if (!path) std::abort();
    void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) std::abort();
    return library;
}
}
extern "C" int cs1_copy_dd(void* dst, const void* src, size_t bytes, void* stream) {
    void* library = real_library();
    using Copy = int (*)(void*, const void*, size_t, void*);
    auto copy = reinterpret_cast<Copy>(dlsym(library, "cs1_copy_dd"));
    if (!copy || copy == &cs1_copy_dd) std::abort();
    int result = copy(dst, src, bytes, stream);
    if (result == 0 && !injected.exchange(true)) {
        pending.store(true);
        std::fputs("Injected device-copy failure after a real queued D2D copy\n", stderr);
        result = 2; // cudaErrorMemoryAllocation, after the real copy was enqueued.
    }
    dlclose(library);
    return result;
}
extern "C" int cs1_stream_sync(void* stream) {
    void* library = real_library();
    using Sync = int (*)(void*);
    auto sync = reinterpret_cast<Sync>(dlsym(library, "cs1_stream_sync"));
    if (!sync || sync == &cs1_stream_sync) std::abort();
    int result = sync(stream);
    if (result == 0 && pending.exchange(false)) {
        const char* path = std::getenv("CUA_S1_TEST_COPY_FAILURE_EVIDENCE");
        if (!path) std::abort();
        FILE* evidence = std::fopen(path, "w");
        if (!evidence) std::abort();
        std::fputs("drained\n", evidence);
        if (std::fclose(evidence) != 0) std::abort();
    }
    dlclose(library);
    return result;
}
