"""GPU regression against the pinned Transformers 5.17 no-FLA implementation.

Run with Python 3.12, torch 2.14.0+cu130, transformers 5.17.0:
  python tests/qwen3_5/test_gdn_fp32.py --library target/release/libqwen3_5_cuda.so
The relative-L2 gate is fixed at 1e-4; repeatability/state isolation is exact.
"""

import argparse
import ctypes
import inspect
import json
import math

import torch
import transformers
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    assert torch.__version__.split("+")[0] == "2.14.0", torch.__version__
    assert transformers.__version__ == "5.17.0", transformers.__version__
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(141)
    lib = ctypes.CDLL(args.library)
    create = lib.cs1_gdn_fp32_create
    create.argtypes, create.restype = [ctypes.c_size_t], ctypes.c_void_p
    destroy = lib.cs1_gdn_fp32_destroy
    destroy.argtypes, destroy.restype = [ctypes.c_void_p], None
    size = lib.cs1_gdn_fp32_workspace_floats
    size.argtypes, size.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_size_t
    run = lib.cs1_gdn_prefill_fp32
    run.argtypes = (
        [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    )
    run.restype = ctypes.c_int
    # Initialize cuBLAS before checking repeated context ownership.
    warm = create(32 << 20)
    assert warm
    destroy(warm)
    torch.cuda.synchronize()
    free_before = torch.cuda.mem_get_info()[0]
    for _ in range(3):
        context = create(32 << 20)
        assert context
        destroy(context)
    torch.cuda.synchronize()
    assert torch.cuda.mem_get_info()[0] == free_before, "context scratch leaked"
    for tokens, heads in [(0, 32), (-1, 32), (65537, 32), (1, 0), (1, 257)]:
        assert size(tokens, heads) == 0
    reference = inspect.unwrap(torch_chunk_gated_delta_rule)
    context = create(32 << 20)
    assert context
    try:
        for tokens, heads, key_heads in [
            (t, 32, 16)
            for t in [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 233, 513, 2049]
        ] + [(129, 4, 2)]:
            q = torch.randn(
                1, tokens, key_heads, 128, device="cuda", dtype=torch.bfloat16
            )
            k = torch.randn_like(q)
            v = torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
            g = -torch.rand(1, tokens, heads, device="cuda") * 0.25
            beta = torch.rand(1, tokens, heads, device="cuda", dtype=torch.bfloat16)
            work = torch.empty(size(tokens, heads), device="cuda", dtype=torch.float32)
            out = torch.empty_like(v)
            stream = torch.cuda.current_stream().cuda_stream
            pointers = [x.data_ptr() for x in (q, k, v, g, beta, out, work)]
            call = [
                context,
                *pointers,
                tokens,
                heads,
                key_heads,
                1 / math.sqrt(128),
                stream,
            ]
            expected, _ = reference(
                q.repeat_interleave(heads // key_heads, dim=2),
                k.repeat_interleave(heads // key_heads, dim=2),
                v,
                g,
                beta,
                use_qk_l2norm_in_kernel=True,
            )
            assert run(*call) == 0
            torch.cuda.synchronize()
            saved = out.clone()
            relative = (
                torch.linalg.vector_norm(out.float() - expected.float())
                / torch.linalg.vector_norm(expected.float())
            ).item()
            assert relative <= 1e-4, (tokens, heads, relative)
            # Poison scratch and run unrelated data before replaying the original input.
            work.fill_(float("nan"))
            changed = torch.randn_like(v)
            changed_call = call.copy()
            changed_call[3] = changed.data_ptr()
            assert run(*changed_call) == 0
            assert run(*call) == 0
            torch.cuda.synchronize()
            assert torch.equal(out, saved), "previous request leaked recurrent state"
            # Invalid shape/scale must fail before a launch or device dereference.
            for index, invalid in [(8, 0), (10, 3), (11, float("nan")), (11, 0.0)]:
                invalid_call = call.copy()
                invalid_call[index] = invalid
                assert run(*invalid_call) != 0
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "heads": heads,
                        "key_heads": key_heads,
                        "relative_l2": relative,
                        "bf16_mismatches": int((saved != expected).sum().item()),
                        "state_reset": True,
                    }
                ),
                flush=True,
            )
    finally:
        destroy(context)


if __name__ == "__main__":
    main()
