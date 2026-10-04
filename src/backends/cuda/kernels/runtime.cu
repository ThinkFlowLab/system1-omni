#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int invalid_argument = 1000;
}

extern "C" {
uint32_t laya_abi_version() { return 1; }

const char* laya_error_string(int code) {
  return code == invalid_argument ? "invalid runtime argument"
                                 : cudaGetErrorString(static_cast<cudaError_t>(code));
}

int laya_set_device(int device) { return cudaSetDevice(device); }

int laya_stream_create(void** stream) {
  if (!stream) return invalid_argument;
  *stream = nullptr;
  cudaStream_t created = nullptr;
  cudaError_t status = cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking);
  if (status == cudaSuccess) *stream = created;
  return status;
}

int laya_alloc(void** p, size_t bytes) {
  if (!p) return invalid_argument;
  *p = nullptr;
  if (!bytes) return invalid_argument;
  void* allocated = nullptr;
  cudaError_t status = cudaMalloc(&allocated, bytes);
  if (status == cudaSuccess) *p = allocated;
  return status;
}

int laya_free(void* p) {
  if (!p) return invalid_argument;
  return cudaFree(p);
}

int laya_upload(void* dst, const void* src, size_t bytes, void* stream) {
  if (!bytes) return 0;
  if (!dst || !src || !stream) return invalid_argument;
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                         static_cast<cudaStream_t>(stream));
}

int laya_download(void* dst, const void* src, size_t bytes, void* stream) {
  if (!bytes) return 0;
  if (!dst || !src || !stream) return invalid_argument;
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                         static_cast<cudaStream_t>(stream));
}

int laya_sync(void* stream) {
  if (!stream) return invalid_argument;
  return cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}

int laya_stream_free(void* stream) {
  if (!stream) return invalid_argument;
  return cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

int laya_capture_begin(void* stream) {
  if (!stream) return invalid_argument;
  return cudaStreamBeginCapture(static_cast<cudaStream_t>(stream),
                                cudaStreamCaptureModeThreadLocal);
}

// Always end capture and destroy the temporary graph, including on failure.
int laya_capture_end(void* stream, void** executable) {
  if (!executable) return invalid_argument;
  *executable = nullptr;
  if (!stream) return invalid_argument;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t created = nullptr;
  cudaError_t status = cudaStreamEndCapture(static_cast<cudaStream_t>(stream), &graph);
  if (status == cudaSuccess && graph)
    status = cudaGraphInstantiate(&created, graph, nullptr, nullptr, 0);
  if (graph) {
    cudaError_t destroyed = cudaGraphDestroy(graph);
    if (status == cudaSuccess) status = destroyed;
  }
  if (status != cudaSuccess || !created) {
    if (created) cudaGraphExecDestroy(created);
    return status != cudaSuccess ? status : invalid_argument;
  }
  *executable = created;
  return 0;
}

int laya_graph_run(void* executable, void* stream) {
  if (!executable || !stream) return invalid_argument;
  return cudaGraphLaunch(static_cast<cudaGraphExec_t>(executable),
                         static_cast<cudaStream_t>(stream));
}

int laya_graph_free(void* executable) {
  if (!executable) return invalid_argument;
  return cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(executable));
}

}
