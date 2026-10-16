// Test-only deterministic capture failure; eager CUDA math remains real.
#include "ops.h"
int cs1_graph_begin(void* stream) { (void)stream; return 2; }
