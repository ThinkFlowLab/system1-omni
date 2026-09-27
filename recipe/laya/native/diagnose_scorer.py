import ctypes, json
from pathlib import Path
import torch
from fast_candidate import make_router
from laya.common import collate_items

router, agent = make_router("fast_no_graph")
m = agent.model
lib = ctypes.CDLL(str(Path("generated/liblaya_cuda.so").resolve()))
ptr = ctypes.c_void_p
lib.laya_blas_create.argtypes = [ctypes.POINTER(ptr), ptr]
lib.laya_linear.argtypes = [
    ptr,
    ptr,
    ptr,
    ptr,
    ptr,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ptr,
]
lib.laya_blas_free.argtypes = [ptr]
stream = ptr(torch.cuda.current_stream().cuda_stream)
handle = ptr()
assert lib.laya_blas_create(ctypes.byref(handle), stream) == 0
case = next(
    c
    for c in json.loads(Path("evidence/model-reference.json").read_text())
    if c["name"] == "short_1"
)
h = (
    torch.frombuffer(
        bytearray(Path("evidence/model/short_1/hidden.f32").read_bytes()),
        dtype=torch.float32,
    )
    .view(case["N"], case["L"], 1024)
    .cuda()
)
qs = case["request"]["questions"]
it = agent._encode_state(
    case["request"]["state"],
    list(qs),
    {k: agent._to_internal(v) for k, v in qs.items()},
)
batch = collate_items([it], agent.tok.pad_token_id)
mk = h[:, batch["marker_pos"][0], :]
with torch.no_grad():
    x = m.scorer[0](mk).reshape(-1, 1024).bfloat16().contiguous()
    for layer, gelu in [(m.scorer[1], True), (m.scorer[3], False)]:
        w = layer.weight.bfloat16().contiguous()
        b = layer.bias.bfloat16().contiguous()
        out = torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=torch.bfloat16)
        assert (
            lib.laya_linear(
                handle,
                ptr(x.data_ptr()),
                ptr(w.data_ptr()),
                ptr(b.data_ptr()),
                ptr(out.data_ptr()),
                x.shape[0],
                w.shape[0],
                w.shape[1],
                0,
                stream,
            )
            == 0
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = layer(x)
        fp = torch.nn.functional.linear(x.float(), w.float(), b.float()).bfloat16()
        separate = torch.nn.functional.linear(x, w, None) + b
        torch.cuda.synchronize()
        print(
            "LAYER",
            w.shape,
            "native/ref unequal",
            torch.count_nonzero(out != ref).item(),
            "max",
            float((out.float() - ref.float()).abs().max()),
            "f32/ref",
            torch.count_nonzero(fp != ref).item(),
            "separate/ref",
            torch.count_nonzero(separate != ref).item(),
            flush=True,
        )
        if not gelu:
            print(
                "native",
                out.tolist(),
                "ref",
                ref.tolist(),
                "f32",
                fp.tolist(),
                "separate",
                separate.tolist(),
                flush=True,
            )
        x = torch.nn.functional.gelu(ref) if gelu else ref
assert lib.laya_blas_free(handle) == 0
