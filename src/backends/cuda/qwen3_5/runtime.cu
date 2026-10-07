// The CUDA runtime calls the Rust side needs, so that it loads one library
// (libqwen3_5_cuda.so) and never links CUDA itself.
#include <cuda_runtime.h>

#include "ops.h"

extern "C" {

uint32_t cs1_abi_version(void) { return CS1_ABI_VERSION; }

const char* cs1_error_string(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }

int cs1_set_device(int device) { return cudaSetDevice(device); }

int cs1_malloc(void** ptr, size_t bytes) { return cudaMalloc(ptr, bytes); }

int cs1_free(void* ptr) { return cudaFree(ptr); }

int cs1_stream_create(void** stream) {
    return cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(stream), cudaStreamNonBlocking);
}

int cs1_stream_destroy(void* stream) { return cudaStreamDestroy(static_cast<cudaStream_t>(stream)); }

int cs1_stream_sync(void* stream) { return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); }

int cs1_upload(void* dst, const void* src, size_t bytes, void* stream) {
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st);
    return e != cudaSuccess ? e : cudaStreamSynchronize(st);
}

int cs1_download(void* dst, const void* src, size_t bytes, void* stream) {
    const cudaStream_t st = static_cast<cudaStream_t>(stream);
    const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, st);
    return e != cudaSuccess ? e : cudaStreamSynchronize(st);
}

int cs1_graph_begin(void* stream) {
    const cudaError_t e = cudaStreamBeginCapture(static_cast<cudaStream_t>(stream), cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) (void)cudaGetLastError();
    return e;
}

int cs1_graph_end(void* stream, void** exec) {
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(static_cast<cudaStream_t>(stream), &graph);
    if (e == cudaSuccess) e = cudaGraphInstantiate(reinterpret_cast<cudaGraphExec_t*>(exec), graph, 0);
    if (graph) cudaGraphDestroy(graph);
    // The returned error is already reported; do not poison the next capture.
    if (e != cudaSuccess) (void)cudaGetLastError();
    return e;
}

int cs1_graph_launch(void* exec, void* stream) {
    const cudaError_t e = cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec), static_cast<cudaStream_t>(stream));
    if (e != cudaSuccess) (void)cudaGetLastError();
    return e;
}

int cs1_graph_destroy(void* exec) { return cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec)); }

}  // extern "C"
