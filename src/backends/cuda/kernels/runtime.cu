#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

extern "C" {
int laya_kernels_init();

int laya_init(void** stream) {
  auto e = cudaSetDevice(0);
  if (e != cudaSuccess)
    return e;
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  if (major != 9)
    return -2;
  e = cudaStreamCreateWithFlags(reinterpret_cast<cudaStream_t*>(stream),
                                cudaStreamNonBlocking);
  if (e != cudaSuccess)
    return e;
  int rc = laya_kernels_init();
  if (rc) {
    cudaStreamDestroy(*reinterpret_cast<cudaStream_t*>(stream));
    *stream = nullptr;
  }
  return rc;
}

const char* laya_error(int code) {
  return code < 0 ? "invalid native CUDA argument or unsupported GPU"
         : code >= 10000 ? "CUDA driver error"
                         : cudaGetErrorString(static_cast<cudaError_t>(code));
}

int laya_alloc(void** p, size_t bytes) {
  return cudaMalloc(p, bytes);
}

int laya_free(void* p) {
  return cudaFree(p);
}

int laya_upload(void* dst, const void* src, size_t bytes, void* stream) {
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                         static_cast<cudaStream_t>(stream));
}

int laya_download(void* dst, const void* src, size_t bytes, void* stream) {
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                         static_cast<cudaStream_t>(stream));
}

int laya_sync(void* stream) {
  return cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}

int laya_stream_free(void* stream) {
  return cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

int laya_capture_begin(void* stream) {
  return cudaStreamBeginCapture(static_cast<cudaStream_t>(stream),
                                cudaStreamCaptureModeThreadLocal);
}

int laya_capture_end(void* stream, void** executable) {
  cudaGraph_t graph = nullptr;
  auto e = cudaStreamEndCapture(static_cast<cudaStream_t>(stream), &graph);
  if (e != cudaSuccess)
    return e;
  e = cudaGraphInstantiate(reinterpret_cast<cudaGraphExec_t*>(executable), graph, 0);
  cudaGraphDestroy(graph);
  return e;
}

int laya_graph_run(void* executable, void* stream) {
  return cudaGraphLaunch(static_cast<cudaGraphExec_t>(executable),
                         static_cast<cudaStream_t>(stream));
}

int laya_graph_free(void* executable) {
  return cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(executable));
}
}
