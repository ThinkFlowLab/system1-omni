// CPU-only ABI fixture: record launches and ownership; do not perform model math.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int capture_active, graph_mode, begins, ends, freed, allocations;
static int stream_value, stream_frees, copy_mode, sync_count;
static void *captured;
void test_graph_mode(int mode) { graph_mode = mode; }
int test_begins(void) { return begins; }
int test_ends(void) { return ends; }
int test_freed(void) { return freed; }
int test_allocations(void) { return allocations; }
int test_stream_frees(void) { return stream_frees; }
void test_copy_mode(int mode) { copy_mode = mode; }
int test_syncs(void) { return sync_count; }

typedef struct Allocation {
  void *pointer;
  size_t bytes;
  struct Allocation *next;
} Allocation;
static Allocation *all_allocations;
static size_t live_bytes, peak_bytes;
size_t test_live_bytes(void) { return live_bytes; }
size_t test_peak_bytes(void) { return peak_bytes; }
void test_reset_peak(void) { peak_bytes = live_bytes; }

int laya_init(void **stream) { *stream = &stream_value; return 0; }
const char *laya_error(int code) { return "fixture CUDA error"; }
int laya_alloc(void **pointer, size_t bytes) {
  allocations++;
  if (capture_active) return 90;
  Allocation *allocation = malloc(sizeof(*allocation));
  if (!allocation) return 1;
  *pointer = calloc(1, bytes);
  if (!*pointer) { free(allocation); return 1; }
  *allocation = (Allocation){*pointer, bytes, all_allocations};
  all_allocations = allocation;
  live_bytes += bytes;
  if (live_bytes > peak_bytes) peak_bytes = live_bytes;
  return 0;
}
int laya_free(void *pointer) {
  Allocation **link = &all_allocations;
  while (*link && (*link)->pointer != pointer) link = &(*link)->next;
  if (*link) {
    Allocation *allocation = *link;
    *link = allocation->next;
    live_bytes -= allocation->bytes;
    free(allocation);
  }
  free(pointer);
  return 0;
}
int laya_upload(void *destination, const void *source, size_t bytes, void *stream) {
  memcpy(destination, source, bytes);
  // Submission may report failure after copying; Rust still must drain it.
  return copy_mode & 1 ? 77 : 0;
}
int laya_download(void *destination, const void *source, size_t bytes, void *stream) {
  memcpy(destination, source, bytes);
  return 0;
}
int laya_sync(void *stream) {
  sync_count++;
  if (capture_active) return 90;
  return copy_mode & 2 ? 78 : 0;
}
int laya_stream_free(void *stream) { stream_frees++; return 0; }
int laya_fill(void **pointers, int batch, int length, int rows, void *stream) {
  if (capture_active) captured = pointers[0];
  else memset(pointers[0], 73, 64);
  return 0;
}

static char trace[1048576];
static size_t used;
void test_reset(void) { used = 0; trace[0] = 0; }
const char *test_trace(void) { return trace; }
static int launch(const char *name, void **pointers, int batch, int length,
                  int rows, int count) {
  // Reserve a complete record before snprintf so long cache tests cannot overflow.
  if (sizeof(trace) - used < 512) return -1;
  used += snprintf(trace + used, sizeof(trace) - used, "%s(%d,%d,%d)",
                   name, batch, length, rows);
  for (int i = 0; i < count; i++)
    used += snprintf(trace + used, sizeof(trace) - used, "/%p", pointers[i]);
  used += snprintf(trace + used, sizeof(trace) - used, "\n");
  return 0;
}
#define KERNEL(name, count) \
  int laya_##name(void **p, int b, int l, int rows, void *stream) { \
    return launch(#name, p, b, l, rows, count); \
  }
KERNEL(embed, 5)
KERNEL(qkv, 4)
KERNEL(rope, 3)
KERNEL(rope_original, 3)
KERNEL(attn_full, 3)
KERNEL(attn_local, 3)
KERNEL(out, 4)
KERNEL(addln, 5)
KERNEL(geglu, 3)
KERNEL(down, 4)
KERNEL(type, 4)
KERNEL(ln_bias, 5)
KERNEL(head_in, 4)
KERNEL(head_out, 4)
KERNEL(addln_bias, 5)
KERNEL(ffn1, 4)
KERNEL(ffn2, 4)
KERNEL(residual, 2)
KERNEL(gather, 5)
KERNEL(features, 4)
#ifndef OMIT_SPECIALIZED_ATTN
KERNEL(attn_full_b1_l512, 3)
KERNEL(attn_full_b4_l512, 3)
KERNEL(attn_local_b1_l512, 3)
KERNEL(attn_local_b4_l512, 3)
#endif
#undef KERNEL

int laya_blas_create(void **pointer, void *stream) {
  *pointer = malloc(1);
  return *pointer ? 0 : 1;
}
int laya_blas_free(void *pointer) { free(pointer); return 0; }
int laya_linear(void *handle, void *input, void *weight, void *bias, void *output,
                int rows, int columns, int width, int gelu, void *stream) {
  memset(output, 0, (size_t)rows * columns * 2);
  return 0;
}

typedef struct { void *target; } TestGraph;
int laya_capture_begin(void *stream) {
  begins++;
  if (capture_active) return 90;
  capture_active = 1;
  captured = 0;
  return 0;
}
int laya_capture_end(void *stream, void **output) {
  ends++;
  capture_active = 0;
  *output = 0;
  if (graph_mode == 2) return 0; // Null-success injection.
  TestGraph *graph = malloc(sizeof(*graph));
  if (!graph) return 1;
  graph->target = captured;
  *output = graph;
  return graph_mode == 1 ? 91 : 0; // Partial ownership with an error.
}
int laya_graph_run(void *pointer, void *stream) {
  if (!pointer) return 90;
  TestGraph *graph = pointer;
  if (graph->target) memset(graph->target, 73, 64);
  return 0;
}
#ifndef OMIT_GRAPH_FREE
int laya_graph_free(void *pointer) { freed++; free(pointer); return 0; }
#endif
