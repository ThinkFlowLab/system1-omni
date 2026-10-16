# CUDA backend contract

`src/backends/cuda/` is shared by several model executors. Each owns its
operations, its own library and its own numerics; this document fixes only the
parts that have to agree for two libraries to be built, distributed and validated
the same way.

Ownership is stated in [the architecture contracts](../../../docs/architecture.md)
and is not restated here, because a restatement drifts: the shared worker runtime
owns processing orchestration, batching policy and request bookkeeping, while
model executors own forward passes, device state and kernel selection. "Model
engine" is the older word for the second of those. A backend does not need
identical internal structures to another backend, and no universal tensor
abstraction is introduced here.

Status: proposed. The checker is [`check_contract.py`](check_contract.py); wiring it
into CI is a separate change, so nothing here is enforced yet.

## Why this exists

Three CUDA efforts are now in flight, each with its own build path:

| Effort | Kernels | Produces | Built by |
| --- | --- | --- | --- |
| Laya native (#14) | TileLang, exported to CUDA | `liblaya_cuda.so` | `tools/export.py` + `tools/build.py` |
| Cua-S1 native (#19) | CUDA C++ | `libqwen3_5_cuda.so` | `build.sh` |
| Cua-S1 multimodal (#12) | Triton / cuTile | (Python) | — |

That is three build systems, three library names and three C ABIs. None of it
conflicts today, because nothing links against anything else yet. It conflicts
as soon as one model reuses another's kernels — which has already happened:
Cua-S1's and Open-Jev's native workers share the Qwen3.5/3.8 executor under
`src/models/qwen3_5/native/`, and `#9` asks for CUDA paths for further models.

`#6` says to extract shared code once two implementations exist. Two exist now.
This document is that extraction, kept to interfaces only.

## 1. Artifact discovery

Every CUDA backend declares itself in one JSON file:

```text
src/backends/cuda/<name>/<name>.backend.json
```

The checker discovers backends by that path, so adding a backend needs no CI
change. Schema:

```json
{
  "name": "qwen3_5",
  "abi_version": 7,
  "status": "validated",
  "sources": ["common.cuh", "mma.cuh", "ops.h", "norm.cu", "elementwise.cu",
              "attention.cu", "gdn_prefill.cu", "gemm.cu", "runtime.cu"],
  "models": ["src/models/cua_s1/native/", "src/models/open_jev/native/"],
  "build": {
    "script": "build.sh",
    "output": "libqwen3_5_cuda.so",
    "default_arch": 89,
    "architectures": [89],
    "min_capability": 80
  },
  "numerics": {
    "precision": "bfloat16",
    "accumulation": "float32",
    "tolerance": { "max_abs": 0.039, "note": "float32 reference, see #11" }
  },
  "reference": {
    "entrypoint": "recipe/cua_s1/check_native.py",
    "note": "pinned upstream FourBModel; kernel tolerances are declared per suite"
  }
}
```

Required keys are `name`, `abi_version`, `status`, `sources` and `build`.
`status` is one of:

- `planned` — directory only; the checker skips everything else.
- `experimental` — builds, but no parity claim yet. `numerics.tolerance` may be
  omitted.
- `validated` — a parity claim is made. `numerics.tolerance` and
  `reference.entrypoint` are required, and the checker fails without them.

`reference.entrypoint` is a repository-relative script that runs on a GPU and
exits non-zero when parity fails. It takes no contract-defined arguments: a
self-hosted runner that has the weights and the GPU runs it directly.

### One backend, several models

`models` lists the model executors that consume this backend, as
repository-relative directories. It is optional, and it exists because reuse is
the point: these kernels serve a Qwen3.5/3.8 prefill, and both the Cua-S1 and the
Open-Jev native worker run their forward pass through them, so one backend
directory serves both. Without this field that sharing is a private arrangement
between PRs and invisible to anyone reading either one. An entry naming a directory
that is not in the tree is a warning, not an error, so a backend can be merged
before its second consumer lands.

### Build script interface

`build.script` is invoked by the compile job as:

```sh
./<build.script> <output-dir> <compute-capability>
```

so it must be executable, accept an output directory as `$1`, and accept a
compute capability as `$2` — defaulting to `build.default_arch` when `$2` is
absent. `#19`'s `build.sh` already has this shape:

```sh
out=${1:?usage: build.sh <output dir> [compute capability]}
arch=${2:-${CUDA_COMPUTE_CAP:-89}}
```

This is a deliberate narrowing, and it is the first place the two in-flight
build paths diverge. `#19` builds from a shell script that emits one library.
`#14` builds through `tools/export.py` and `tools/build.py`, which are Python
programs driven by a TileLang export step and take a bundle directory, not an
output directory plus an architecture. A backend of that shape satisfies the
contract with a small `build.sh` wrapper that documents the real invocation
rather than by CI growing a second code path. Making that wrapper the required
entry point keeps CI one path and makes "how do I build this backend" answerable
from the manifest alone.

## 2. C ABI

A backend library exports a flat `extern "C"` interface and is loaded at run
time, so a Rust engine builds without a CUDA toolkit. `#19`'s `ops.h` is the
reference for the shape. The contract fixes four points:

1. **One version symbol per library, and the manifest repeats it.** A backend
   defines `<PREFIX>_ABI_VERSION` in a header and exports
   `uint32_t <prefix>_abi_version(void)` returning it; a loader refuses a library
   whose value it does not know. The manifest's `abi_version` is **that number**,
   not a version of this document, and the checker reads the macro out of the
   declared sources and requires the two to agree. `#19`'s `ops.h` says
   `CS1_ABI_VERSION 7` today, so a `qwen3_5` manifest declares 7.

   Two backends may declare different values. They are independent libraries, and
   the repository layout says CUDA and Metal implementations need not share
   internal structure; forcing one number across models would invent a coupling
   nothing needs. What must hold is that each manifest matches its own header.
2. **Errors are `int`, not exceptions.** Every entry point returns `0` on
   success. CUDA runtime errors are returned as `cudaError_t` values; anything
   the library defines itself starts at `1000`. Every library exports
   `const char* <prefix>_error_string(int)`.
3. **Work is queued on a caller-supplied stream.** Operations take a
   `cudaStream_t` as their last argument and must not synchronize internally, so
   that a caller can capture them into a CUDA Graph. Allocations, copies, stream
   creation and graph capture are exported by the library too, so a caller needs
   no direct CUDA linkage.
4. **GEMM algorithm choice is explicit and reproducible.** A tuned plan is
   exported and imported rather than re-tuned at load. A plan that reduces
   split-K in place must be refused, so a given plan always produces the same
   result.

Point 3 is what makes two model libraries composable: a model executor that
already owns a stream and a captured graph can call into either library.

### Runtime symbol sharing

`#19`'s runtime block (`malloc`, `free`, `stream_create`, `stream_sync`,
`upload`, `download`, `graph_begin`, `graph_end`, `graph_launch`,
`graph_destroy`, `device_info`, `set_device`) is the same work every backend
needs. Implementing it per library means two libraries loaded into one process
each carry a copy, and a graph captured through one cannot be launched through
the other.

Extracting that set into one shared header is a follow-up, deliberately not part
of this change: there is only one backend today, so the shape of the shared
surface would be guesswork. It becomes worth doing when a second backend
actually needs to be loaded alongside the first, and the interface can be
written against two real callers instead of an imagined one.

## 3. Numerics

Bit-exactness is not claimed anywhere, so the contract records what varies
instead of leaving it to be discovered in a parity failure:

- **Rounding points.** Where each kernel rounds to the storage dtype is part of
  the kernel's documentation. `#19`'s `qwen3_5/README.md` is the model: it names
  the kernels that round where the Transformers reference rounds, and the
  attention and Gated DeltaNet paths that keep bfloat16 intermediates as
  FlashAttention and flash-linear-attention do.
- **Accumulation.** Float32 unless documented otherwise.
- **Tolerance.** Declared before comparison, per suite, in
  `numerics.tolerance`. A single end-to-end number is not enough: an end-to-end
  check cannot localize a failure to one kernel.
- **Architecture.** `build.architectures` lists the compute capabilities the
  library is built for. A kernels file that requires a newer capability than the
  build script's default is a contract error — `#19`'s `mma.cuh` and
  `gdn_prefill.cu` both say "sm_80 and later" in their first line, while `#14`
  builds sm_90a only.
- **Kernel requirement.** `build.min_capability` states the oldest compute
  capability the kernels actually compile for. It exists because that fact is
  currently only in prose — `#19`'s `gdn_prefill.cu` and `mma.cuh` both say
  "sm_80 and later" in a header comment, and its README says "Tensor-core kernels
  need sm_80 or newer". One integer turns that into something CI checks: every
  entry in `build.architectures` must be at least `build.min_capability`, so a
  library cannot claim a target its own kernels reject. The field is optional;
  a backend that omits it declares no verified floor, which is honest but not
  checked.

## 4. Validation

Two tiers, because they have different requirements.

**Tier 1 — contract check (no GPU).** `check_contract.py` reads the manifests and
the build scripts and checks: the schema, that every declared source exists, that
a build script declares the architectures the manifest claims and writes the
library the manifest names, that `abi_version` matches the `<PREFIX>_ABI_VERSION`
macro the declared sources define, and that a `validated` backend declares both a
tolerance and a reference entrypoint. It runs on a stock runner and needs no CUDA
toolkit. This is the tier implemented in this change.

It does not compile CUDA and does not prove numerics.

**Tier 2 — compile (no GPU).** Building each backend with `nvcc` in a CUDA
container, so a kernel that stops compiling cannot land. The interface it relies
on is the one Tier 1 already checks: `./<build.script> <output-dir>
<compute-capability>`, producing exactly the declared library. Not yet wired up;
it needs a container image and belongs with the rest of the CI wiring rather than
with the checker.

Note what a build check does and does not prove. It shows the sources compile and
that a file of the declared name appears. It does not prove the binary honoured
`build.architectures`: `#19`'s `build.sh` advertises that it "also embeds PTX",
and nothing reads the artifact to confirm which targets are inside. Inspecting
that would need `cuobjdump`, which is a reasonable follow-up rather than
something this change claims to do.

**Tier 3 — GPU parity (self-hosted).** Running each validated backend's
`reference.entrypoint` on a real GPU, and reporting kernel-level and end-to-end
parity separately. Contributors with a card can attach their own runner to their
own fork, which is how `#19` (RTX 6000 Ada, sm_89), `#12` (RTX 4090) and `#14`
(H800, sm_90a) can each be checked on the hardware they were measured on. Not yet
wired up.

The split is the point. A CPU-only workflow that claims to validate CUDA is worse
than no workflow, which is the gap `#14`'s own validation report already records:
today's CI checks every Rust feature and compiles no CUDA at all. Tier 1 removes
the part of that gap which does not need hardware; Tiers 2 and 3 need hardware and
are therefore separate.

## Reporting

Kernel-level and end-to-end results stay separate, and both report the hardware,
driver, CUDA version, compute capability, source revision and tolerance that
applied. Load, warmup and warm latency are reported separately. A run that does
not establish an improvement reports that, rather than an acceleration claim.
