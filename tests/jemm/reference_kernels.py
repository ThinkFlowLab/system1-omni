#!/usr/bin/env python3
"""GPU regression for the optional JEMM arithmetic and capture-safe workspaces.

Requires the pinned Torch/FLA environment and a cuDNN-enabled CUDA library.
Run only under exclusive GPU admission; this is correctness, not benchmarking.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261011)
    torch.cuda.set_device(0)
    execution_stream = torch.cuda.Stream()
    torch.cuda.set_stream(execution_stream)
    lib = ct.CDLL(str(args.library.resolve()))
    ptr, integer = ct.c_void_p, ct.c_int
    signatures = {
        "gdn_workspace_floats": ([integer, integer], ct.c_size_t),
        "gdn_prefill": ([ptr] * 7 + [integer] * 3 + [ct.c_float, ptr], integer),
        "language_attention_workspace_floats": ([integer, integer], ct.c_size_t),
        "vision_attention_workspace_floats": ([integer, integer], ct.c_size_t),
        "attention": ([ptr, ptr, ptr, integer, ptr, ptr, integer, integer, integer, ct.c_float, ptr, ptr], integer),
        "vision_attention": ([ptr] * 4 + [integer] * 3 + [ptr, ptr], integer),
        "rms_norm": ([ptr] * 3 + [integer, integer, ct.c_float, ptr], integer),
        "patch_create": ([integer, ptr], ptr),
        "patch": ([ptr] * 4, integer),
        "patch_destroy": ([ptr], None),
    }
    api = {}
    for name, (arguments, result) in signatures.items():
        fn = getattr(lib, "cs1_reference_" + name)
        fn.argtypes, fn.restype = arguments, result
        api[name] = fn
    stream = ptr(torch.cuda.current_stream().cuda_stream)
    p = lambda tensor: ptr(tensor.data_ptr())
    rand = lambda shape: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    records = []

    def verify(name, actual, expected, invoke, max_abs=0.0, relative_l2=0.0):
        assert invoke() == 0, name
        torch.cuda.synchronize()
        difference = actual.float() - expected.float()
        record = {"name": name, "elements": actual.numel(),
                  "different": int((actual != expected).sum()),
                  "max_abs": float(difference.abs().max()),
                  "relative_l2": float(difference.norm() / expected.float().norm().clamp_min(1e-30))}
        assert record["max_abs"] <= max_abs and record["relative_l2"] <= relative_l2, record
        eager = actual.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            assert invoke() == 0, name
        actual.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual, eager), f"{name}: capture/replay differs from eager"
        record["graph_replay_exact"] = True
        records.append(record)
        print(json.dumps(record), flush=True)

    for tokens in (63, 278, 475):
        heads, dim = 48, 128
        q, k, v = (rand((1, tokens, heads, dim)) for _ in range(3))
        g = -torch.rand((1, tokens, heads), device="cuda", dtype=torch.float32)
        beta = torch.sigmoid(rand((1, tokens, heads)))
        expected, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        out = torch.empty_like(v)
        workspace = torch.empty(api["gdn_workspace_floats"](tokens, heads), device="cuda", dtype=torch.float32)
        invoke = lambda: api["gdn_prefill"](p(q), p(k), p(v), p(g), p(beta), p(out), p(workspace), tokens, heads, heads, dim ** -0.5, stream)
        verify(f"gdn-{tokens}", out, expected, invoke)

    for tokens in (63, 278, 475):
        heads, kv_heads, dim = 24, 4, 256
        q = rand((tokens, heads, dim))
        k, v = (rand((tokens, kv_heads, dim)) for _ in range(2))
        gate = rand(q.shape)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            attention = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
                is_causal=True, enable_gqa=True).squeeze(0).transpose(0, 1).contiguous()
        expected = attention * torch.sigmoid(gate)
        out = torch.empty_like(q)
        workspace = torch.empty(api["language_attention_workspace_floats"](512, heads), device="cuda", dtype=torch.float32)
        invoke = lambda: api["attention"](p(q), p(k), p(v), kv_heads * dim, p(gate), p(out), tokens, heads, kv_heads, dim ** -0.5, p(workspace), stream)
        verify(f"attention-{tokens}", out, expected, invoke)

    for rows in (1, 278):
        x, w = rand((rows, 5120)), rand((5120,))
        expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * (1 + w.float())).bfloat16()
        out = torch.empty_like(x)
        verify(f"rms-{rows}", out, expected,
               lambda: api["rms_norm"](p(x), p(w), p(out), rows, 5120, 1e-6, stream))

    for tokens in (256, 512):
        heads, dim = 16, 72
        q, k = rand((tokens, heads, dim)), rand((tokens, heads, dim))
        qkv = rand((tokens, 3, heads, dim))
        v = qkv[:, 2]
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            expected = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1).contiguous()
        out = torch.empty_like(q)
        floats = api["vision_attention_workspace_floats"](512, heads)
        workspace = torch.empty(512 * heads * 80 * 4 * 2 + floats * 4, device="cuda", dtype=torch.uint8)
        verify(f"vision-attention-{tokens}", out, expected,
               lambda: api["vision_attention"](p(q), p(k), p(v), p(out), tokens, heads, dim, p(workspace), stream))

    x, w = rand((256, 3, 2, 16, 16)), rand((1152, 3, 2, 16, 16))
    expected = F.conv3d(x, w, stride=(2, 16, 16)).reshape(256, 1152)
    out = torch.empty_like(expected)
    plan = api["patch_create"](256, stream)
    assert plan, "cuDNN patch plan creation"
    try:
        verify("patch-convolution", out, expected, lambda: api["patch"](plan, p(x), p(w), p(out)))
    finally:
        torch.cuda.synchronize()
        api["patch_destroy"](plan)
    args.report.write_text(json.dumps({"passed": True, "records": records,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
