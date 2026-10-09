"""Build-only TileLang -> CUDA export. Runtime needs CUDA, not Python/TVM/Torch.

Host argument stacks are inspected, including dynamic TMA extents and strides.
Unknown symbols/launch layouts fail generation instead of guessing an ABI.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import tilelang
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
import laya_tilelang as kernels

QKV_DISPATCH = """extern "C" int laya_qkv(void** p,int B,int L,int M,cudaStream_t stream) {
  if(B==1 && L<=64)return laya_qkv_short(p,B,L,M,stream);
  return laya_qkv_full(p,B,L,M,stream);
}
"""

OUT_DISPATCH = """extern "C" int laya_out(void** p,int B,int L,int M,cudaStream_t stream) {
  if(B==1 && L<=64)return laya_out_short(p,B,L,M,stream);
  return laya_out_full(p,B,L,M,stream);
}
"""

SLOT = re.compile(
    r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_(?:int64|ptr)\) = (.*);"
)
CALL = re.compile(
    r"TVMFFIFunctionCall\((\w+?)_packed, \(TVMFFIAny\*\) stack_ffi_any, (\d+),"
)


def host_calls(kernel):
    slots = {}
    calls = []
    for line in kernel.get_host_source().splitlines():
        match = SLOT.search(line)
        if match:
            slots[int(match[1])] = match[2]
            continue
        match = CALL.search(line)
        if match:
            if match[1] in ("__tvm_tensormap_create_tiled", "main_kernel"):
                values = [slots.get(i) for i in range(int(match[2]))]
                if None in values:
                    raise ValueError(("missing argument", match[1], values))
                calls.append((match[1], values))
            slots = {}
    return calls


def integer(value):
    value = value.replace("(int64_t)", "").replace("(", "").replace(")", "")
    return int(value)


def patch_geglu_static_registers(body):
    """Remove fixed register reservations only from the accepted BN32 lowering."""
    expected = "1ed96990542c07b58dca419dafcc45c6163bb9b6fff04ffca5ecedb2a34e921c"
    if hashlib.sha256(body.encode()).hexdigest() != expected:
        raise ValueError("long GEGLU BN32 lowering changed")
    for call in (
        "    tl::warpgroup_reg_dealloc<24>();\n",
        "    tl::warpgroup_reg_alloc<240>();\n",
    ):
        if body.count(call) != 1:
            raise ValueError("long GEGLU register calls changed")
        body = body.replace(call, "", 1)
    return body


def export(name, kernel):
    source = kernel.get_kernel_source()
    signature = re.search(r"void main_kernel\((.*?)\);", source, re.S)[1]
    params = [param.strip() for param in signature.split(",")]
    names = [param.split()[-1].lstrip("*") for param in params]
    bindings = []
    for index, param in enumerate(kernel.prim_func.params):
        buffer = kernel.prim_func.buffer_map[param]
        ctype = {
            "bfloat16": "bfloat16_t",
            "float32": "float",
            "int32": "int",
            "int64": "int64_t",
        }[str(buffer.dtype)]
        bindings.append(f"  auto* {buffer.name}=static_cast<{ctype}*>(p[{index}]);")

    descriptors = []
    launch = None
    for callee, args in host_calls(kernel):
        if callee == "main_kernel":
            launch = args
            continue
        variable, dtype, rank, tensor = args[:4]
        rank_value = integer(rank)
        dtype_value = integer(dtype)
        if dtype_value not in (7, 9) or not 1 <= rank_value <= 5:
            raise ValueError(("unsupported TMA format", name, dtype, rank))
        dtype_enum = {
            7: "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
            9: "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
        }[dtype_value]
        dims = args[4 : 4 + rank_value]
        strides = args[4 + rank_value : 4 + 2 * rank_value]
        box = args[4 + 2 * rank_value : 4 + 3 * rank_value]
        steps = args[4 + 3 * rank_value : 4 + 4 * rank_value]
        interleave, swizzle, l2, oob = map(integer, args[4 + 4 * rank_value :])
        if (
            integer(strides[0]) != {7: 4, 9: 2}[dtype_value]
            or interleave != 0
            or oob != 0
            or swizzle not in range(4)
            or l2 not in range(4)
        ):
            raise ValueError("unsupported TMA layout")
        descriptors.append(
            f"""  alignas(64) CUtensorMap {variable};
  {{ uint64_t dims[]={{{','.join('static_cast<uint64_t>('+v+')' for v in dims)}}}, strides[]={{{','.join('static_cast<uint64_t>('+v+')' for v in strides[1:])}}};
     uint32_t box[]={{{','.join(box)}}}, steps[]={{{','.join(steps)}}};
     CUresult rc=cuTensorMapEncodeTiled(&{variable},{dtype_enum},{rank_value},{tensor},dims,strides,box,steps,
       CU_TENSOR_MAP_INTERLEAVE_NONE,static_cast<CUtensorMapSwizzle>({swizzle}),static_cast<CUtensorMapL2promotion>({l2}),CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
     if(rc!=CUDA_SUCCESS)return 10000+static_cast<int>(rc); }}"""
        )

    if launch is None:
        raise ValueError("no launch")
    # Scalar/pointer entries are exactly the recovered device signature order.
    args = launch[: len(params)]
    tail = launch[len(params) :]
    for parameter, value in zip(names, args):
        if parameter != value and parameter not in ("M", "B", "L"):
            raise ValueError(("argument changed", parameter, value))
    if (
        len(tail) >= 4
        and integer(tail[-2]) == 1
        and integer(tail[-3]) == 1
        and integer(tail[-1]) > 1
    ):
        grid = tail[:-4]
        block = list(map(integer, tail[-4:-1]))
        shared_memory = integer(tail[-1])
    else:
        grid = tail[:-3]
        block = list(map(integer, tail[-3:]))
        shared_memory = 0
    if not 1 <= len(grid) <= 3 or block[1:] != [1, 1] or shared_memory > 227 * 1024:
        raise ValueError(("launch", tail))
    if len(grid) < 3:
        grid += ["1"] * (3 - len(grid))

    symbol = "laya_" + name + "_kernel"
    body = source[source.index('extern "C" __global__') :].replace(
        "main_kernel", symbol
    )
    if name == "geglu":
        body = patch_geglu_static_registers(body)
    # TileLang reuses Q_s for O_s. Its generated wait is inside the key loop,
    # so zero-key rows can overwrite Q_s while the asynchronous Q load is live.
    # Wait before entering producer/consumer branches, including zero iterations.
    # Match the lowered structure strictly; do not silently patch a new lowering.
    if name.startswith("attn_"):
        q_load = re.search(r"tl::tma_load\(QKV_desc, mbarrier\[(\d+)\].*?Q_s.*?;", body)
        if not q_load:
            raise ValueError("attention Q TMA load changed")
        end = body.index("__syncthreads();", q_load.end()) + len("__syncthreads();")
        body = (
            body[:end]
            + f"\n  mbarrier[{q_load[1]}].wait(0); // Q load must complete even with no valid keys.\n"
            + body[end:]
        )
    preamble = source[: source.index('extern "C" __global__')]

    # Never inject unknown identifiers from a compiler expression into the wrapper.
    allowed = (
        set(names)
        | {"M", "B", "L", "int64_t"}
        | {
            str(kernel.prim_func.buffer_map[param].name)
            for param in kernel.prim_func.params
        }
    )
    for _, values in host_calls(kernel):
        for value in values:
            for identifier in re.findall(r"\b[A-Za-z_]\w*\b", value):
                if identifier not in allowed and not identifier.endswith("_desc"):
                    raise ValueError(("unknown host symbol", identifier))
    dispatch = ""
    if name == "geglu":
        dispatch = "  if(B==1 && M<=64)return laya_geglu_short(p,B,L,M,stream);\n"
    wrapper_name = {"qkv": "qkv_full", "out": "out_full"}.get(name, name)
    wrapper = f"""extern "C" int laya_{wrapper_name}(void** p,int B,int L,int M,cudaStream_t stream) {{
  if(B<1 || B>16 || L<16 || L>512 || L%16 || M!=B*L)return -1;
{dispatch}{chr(10).join(bindings)}
{chr(10).join(descriptors)}
  {symbol}<<<dim3({','.join(grid)}),dim3({','.join(map(str,block))}),{shared_memory},stream>>>({','.join(args)});
  return static_cast<int>(cudaGetLastError());
}}
"""
    init = ""
    if shared_memory >= 49152:
        init = f"if(auto e=cudaFuncSetAttribute({symbol},cudaFuncAttributeMaxDynamicSharedMemorySize,{shared_memory});e!=cudaSuccess)return static_cast<int>(e);"
    metadata = {
        "name": name,
        "params": names,
        "block": block,
        "smem": shared_memory,
        "grid": grid,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "emitted_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "zero_key_wait": name.startswith("attn_"),
        "host_calls": host_calls(kernel),
    }
    return preamble, body + wrapper, init, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--rope-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "kernels/rope_selected.py",
    )
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("rope_selected", args.rope_source)
    rope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rope)
    exports = {
        "rope": rope.build(16, 64, 1, 8),
        "rope_original": kernels.rope_kernel(16, 64),
        "qkv": kernels.gemm_kernel(3072, 1024),
        "qkv_short": kernels.gemm_kernel(3072, 1024, bn=64),
        "attn_full": kernels.attn_kernel(None, None, 16, 64),
    }
    if not args.probe_only:
        exports.update(
            {
                "out": kernels.gemm_kernel(1024, 1024),
                "out_short": kernels.gemm_kernel(1024, 1024, bm=64, bn=64),
                "geglu_short": kernels.gemm_geglu_kernel(2624, 1024, bn=32),
                "geglu": kernels.gemm_geglu_kernel(2624, 1024, bn=32),
                "down": kernels.gemm_kernel(1024, 2624, bm=64, bn=64),
                "addln": kernels.add_ln_kernel(1024),
                "addln_bias": kernels.add_ln_kernel(1024, bias=True),
                "ln_bias": kernels.add_ln_kernel(1024, residual=False, bias=True),
                "head_in": kernels.gemm_kernel(3072, 1024, bias=True),
                "head_out": kernels.gemm_kernel(1024, 1024, bias=True),
                "ffn1": kernels.gemm_kernel(4096, 1024, bias=True, act="relu"),
                "ffn2": kernels.gemm_kernel(1024, 4096, bias=True),
                "attn_local": kernels.attn_kernel(None, None, 16, 64, window=64),
            }
        )
        for batch in (1, 4):
            for label, window in [("full", 0), ("local", 64)]:
                exports[f"attn_{label}_b{batch}_l512"] = kernels.attn_kernel(
                    batch, 512, 16, 64, window=window
                )

    preambles = []
    bodies = []
    inits = []
    metadata = []
    for name, kernel in exports.items():
        preamble, body, init, details = export(name, kernel)
        preambles.append(preamble)
        bodies.append(body)
        inits.append(init)
        metadata.append(details)
    # Only one copy of debug helper definitions. Other headers carry include guards.
    preamble = "\n".join(
        dict.fromkeys(
            line
            for block in preambles
            for line in block.splitlines()
            if line.startswith("#include <tl_templates")
        )
    )
    code = (
        "#include <cuda.h>\n#include <cuda_runtime.h>\n"
        + preamble
        + "\n"
        + "\n".join(bodies)
        + QKV_DISPATCH
        + ("" if args.probe_only else OUT_DISPATCH)
        + '\nextern "C" int laya_kernels_init(){'
        + "".join(inits)
        + "return 0;}\n"
    )
    (args.output / "generated.cu").write_text(code)
    manifest = {
        "tilelang_version": tilelang.__version__,
        "kernels": metadata,
        "nvcc_flags": [
            "-std=c++20",
            "-gencode=arch=compute_90a,code=sm_90a",
            "--use_fast_math",
            "-DENABLE_BF16",
        ],
        "include_dirs": [str(TILELANG_TEMPLATE_PATH), str(CUTLASS_INCLUDE_DIR)],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("EXPORTED", len(exports), flush=True)


if __name__ == "__main__":
    main()
