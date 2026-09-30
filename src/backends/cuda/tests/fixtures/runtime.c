#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LAYA_TEST_ABI
#define LAYA_TEST_ABI 1
#endif

enum {
  invalid_argument = 1000,
  mode_create_error = 1,
  mode_copy_error = 2,
  mode_sync_error = 3,
  mode_copy_and_sync_error = 4,
  mode_alloc_error = 5
};

typedef struct { int device, live; } Stream;
typedef struct { int device; unsigned char data[]; } Allocation;
static Stream streams[16];
static int device = -1, mode, next_stream, live;
static char trace[4096], error[64];
static size_t trace_len, pending_bytes;
static void *pending_dst;
static const void *pending_src;

static void record(const char *name) {
  trace_len += (size_t)snprintf(trace + trace_len, sizeof(trace) - trace_len,
                              "%s%d ", name, device);
}
static Allocation *allocation(void *p) {
  return (Allocation *)((unsigned char *)p - offsetof(Allocation, data));
}
static int valid_stream(void *p) {
  Stream *s = p;
  return s && s->live && s->device == device;
}
void laya_test_mode(int value) { mode = value; }
const char *laya_test_trace(void) { return trace; }
int laya_test_live(void) { return live; }
uint32_t laya_abi_version(void) { return LAYA_TEST_ABI; }
const char *laya_error_string(int code) {
  snprintf(error, sizeof(error), "fixture-error-%d", code);
  return error;
}
int laya_set_device(int value) {
  if (value < 0) return 11;
  device = value;
  record("device");
  return 0;
}
int laya_stream_create(void **out) {
  record("create");
  if (mode == mode_create_error) return 23;
  Stream *s = &streams[next_stream++];
  *s = (Stream){device, 1};
  *out = s;
  live++;
  return 0;
}
int laya_alloc(void **out, size_t bytes) {
  record("alloc");
  if (mode == mode_alloc_error) return 31;
  Allocation *a = calloc(1, sizeof(*a) + bytes);
  if (!a) return 32;
  a->device = device;
  *out = a->data;
  live++;
  return 0;
}
int laya_free(void *p) {
  if (!p) return invalid_argument;
  record("free");
  if (allocation(p)->device != device || pending_bytes) return 91;
  free(allocation(p));
  live--;
  return 0;
}
static int copy(void *dst, const void *src, size_t bytes, void *stream) {
  if (!valid_stream(stream)) return 91;
  pending_dst = dst;
  pending_src = src;
  pending_bytes = bytes;
  return mode == mode_copy_error || mode == mode_copy_and_sync_error ? 41 : 0;
}
int laya_upload(void *dst, const unsigned char *src, size_t bytes, void *stream) {
  record("upload");
  if (allocation(dst)->device != device) return 91;
  return copy(dst, src, bytes, stream);
}
#ifndef LAYA_TEST_NO_DOWNLOAD
int laya_download(unsigned char *dst, void *src, size_t bytes, void *stream) {
  record("download");
  if (allocation(src)->device != device) return 91;
  return copy(dst, src, bytes, stream);
}
#endif
int laya_sync(void *stream) {
  record("sync");
  if (!valid_stream(stream)) return 91;
  if (pending_bytes) memcpy(pending_dst, pending_src, pending_bytes);
  pending_bytes = 0;
  return mode == mode_sync_error || mode == mode_copy_and_sync_error ? 42 : 0;
}
int laya_stream_free(void *stream) {
  record("destroy");
  if (!valid_stream(stream) || pending_bytes) return 91;
  ((Stream *)stream)->live = 0;
  live--;
  return 0;
}
