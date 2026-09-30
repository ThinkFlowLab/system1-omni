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
}
