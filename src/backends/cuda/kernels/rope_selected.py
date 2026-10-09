"""Retile official in-place RoPE; arithmetic, precision and layout stay unchanged.

QKV is contiguous BF16 [M, 3*H*Dh], packed (q|k|v)(head)(dim). Cos/Sin
are contiguous FP32 [L, Dh/2], L > 0. Each row uses position r % L.
Q/K pairs are loaded in FP32, rotated in the official operation order and
rounded to BF16; V is never read or written. Inputs must be non-overlapping
CUDA tensors on one device. Dynamic M/L and partial row/head tiles are supported.
Only launch geometry changes. Numerical/performance acceptance is external.
"""

import tilelang
import tilelang.language as T
import torch

DT, ACC = "bfloat16", "float"
FAST = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
VARIANTS = {"r1_h4": (1, 4), "r2_h4": (2, 4), "r1_h8": (1, 8)}


@tilelang.jit(pass_configs=FAST)
def _kernel(H, Dh, rows, heads):
    M, L = T.dynamic("M"), T.dynamic("L")
    half = Dh // 2

    @T.prim_func
    def main(QKV: T.Tensor((M, 3 * H * Dh), DT),
             Cos: T.Tensor((L, half), ACC), Sin: T.Tensor((L, half), ACC)):
        with T.Kernel(T.ceildiv(M, rows), T.ceildiv(2 * H, heads),
                      threads=128) as (bx, by):
            for i, c in T.Parallel(rows, heads * half):
                r = bx * rows + i
                hh = by * heads + c // half
                d = c % half
                if r < M:
                    if hh < 2 * H:
                        pos = r % L
                        c0 = hh * Dh + d
                        c1 = c0 + half
                        x0 = T.cast(QKV[r, c0], ACC)
                        x1 = T.cast(QKV[r, c1], ACC)
                        cs = Cos[pos, d]
                        sn = Sin[pos, d]
                        QKV[r, c0] = T.cast(x0 * cs - x1 * sn, DT)
                        QKV[r, c1] = T.cast(x1 * cs + x0 * sn, DT)
    return main


def build(H, Dh, rows, heads):
    """Build a generic positive-H, even-Dh kernel; measured target is H=16,Dh=64."""
    values = (H, Dh, rows, heads)
    if any(type(value) is not int or value <= 0 for value in values) or Dh % 2:
        raise ValueError("H/rows/heads must be positive integers; Dh must be positive and even")
    return _kernel(H, Dh, rows, heads)


class InstalledRoPE:
    """Count host calls during capture only, never GPU graph replays.

    capture_calls does not prove successful graph construction or replay; use
    the caller's graph inventory and a GPU trace to establish those separately.
    """

    def __init__(self, kernel, H, Dh, variant, rows, heads, original):
        self.kernel = kernel
        self.original = original
        self.metadata = {"variant": variant, "H": H, "Dh": Dh,
                         "rows": rows, "heads": heads, "threads": 128,
                         "fast_math": True, "counter_scope": "host_calls_during_capture"}
        self.capture_calls = 0
        self.capture_shapes = {}

    def __call__(self, qkv, cos, sin):
        H, Dh = self.metadata["H"], self.metadata["Dh"]
        if (qkv.ndim != 2 or qkv.shape[1] != 3 * H * Dh or qkv.shape[0] <= 0
                or cos.ndim != 2 or cos.shape[0] <= 0 or cos.shape[1] != Dh // 2
                or tuple(sin.shape) != tuple(cos.shape)):
            raise ValueError("RoPE requires QKV[M,3*H*Dh] and Cos/Sin[L,Dh/2], M/L > 0")
        if (qkv.dtype != torch.bfloat16 or cos.dtype != torch.float32
                or sin.dtype != torch.float32 or not qkv.is_cuda
                or cos.device != qkv.device or sin.device != qkv.device
                or not all(t.is_contiguous() for t in (qkv, cos, sin))):
            raise ValueError("RoPE requires contiguous CUDA BF16 QKV and FP32 Cos/Sin on one device")
        capturing = torch.cuda.is_current_stream_capturing()
        result = self.kernel(qkv, cos, sin)
        if capturing:
            self.capture_calls += 1
            M, L = int(qkv.shape[0]), int(cos.shape[0])
            key = f"M={M},L={L}"
            entry = self.capture_shapes.setdefault(key, {
                "M": M, "L": L,
                "grid": [(M + self.metadata["rows"] - 1) // self.metadata["rows"],
                         (2 * H + self.metadata["heads"] - 1) // self.metadata["heads"], 1],
                "block": [128, 1, 1], "capture_calls": 0,
            })
            entry["capture_calls"] += 1
        return result


def install(fast, variant):
    """Replace FastLaya RoPE before any graph is built; return capture metadata.

    Call only on an idle FastLaya instance. No existing graph is invalidated or
    silently reused with a different kernel. Compilation failures leave it intact.
    """
    if fast.graphs:
        raise ValueError("Install RoPE on a fresh FastLaya instance with no captured graphs")
    if (fast.H, fast.Dh) != (16, 64):
        raise ValueError("This integration experiment supports only H=16, Dh=64")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown RoPE variant: {variant}")
    if isinstance(fast._rope_k, InstalledRoPE):
        raise ValueError("Restore the previous RoPE candidate before installing another")
    rows, heads = VARIANTS[variant]
    candidate = InstalledRoPE(build(fast.H, fast.Dh, rows, heads), fast.H,
                              fast.Dh, variant, rows, heads, fast._rope_k)
    fast._rope_k = candidate
    return candidate


def restore(fast):
    """Restore the exact saved official callable (or None for official lazy build).

    Existing captured graphs cannot be patched: use a fresh FastLaya instance
    for paired runs, or explicitly dispose of the caller-owned graphs first.
    """
    if fast.graphs:
        raise ValueError("Cannot restore RoPE while captured graphs still reference the candidate")
    if not isinstance(fast._rope_k, InstalledRoPE):
        raise ValueError("No RoPE candidate is installed")
    fast._rope_k = fast._rope_k.original
