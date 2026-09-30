# LAYA model engine

LAYA is the first planned System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

The `omni-laya` crate currently reads and checks the English Laya 0.3.20 checkpoint. `Config::load` validates the architecture and temperatures; `Weights` checks tensor names and shapes and converts FP32, FP16 and BF16 values. `checkpoint_tensors()` lists the 206 expected tensors. Each backend chooses its own storage precision.

Keep checkpoint files unchanged while `Weights` holds a read-only memory mapping. The eager encoder is described below; complete request-to-result inference is not yet implemented.

## CPU checks

The normal workspace tests cover configuration errors, malformed tensors, inventory mismatches and conversion boundaries without downloading weights.

To check the complete checkpoint, use `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and a Python environment with PyTorch, safetensors and NumPy:

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
export LAYA_WEIGHT_ORACLE=/tmp/laya-weight-oracle.json
python recipe/laya/native/export_weights.py "$LAYA_CHECKPOINT" "$LAYA_WEIGHT_ORACLE"
cargo test --release --locked -p omni-laya --test weights -- --ignored
```

These two CPU tests check all 206 tensor names and shapes, 618 conversion hashes, and the legacy temperature buffer. The normal CI job skips them because it does not download the full checkpoint.

## GPU weight residency

`ResidentWeights::upload(&cuda, &weights)` validates the checkpoint inventory and
uploads the weights once. Embeddings use FP16; encoder norms, head norms and biases,
and the scorer input norm use FP32; other weights use BF16. Layouts stay unchanged.
The legacy `temperature` buffer is validated but not uploaded.

`get(name)` returns the resident buffer. `bytes()` reports weight allocations only,
excluding CUDA context and allocator overhead. Buffers keep their CUDA context alive
after the caller drops the source mapping or `Cuda`. Failed loads release partial
allocations. Workspace, rotary tables and inference are separate modules.

The ignored `real_checkpoint_residency_matches_torch` test requires
`LAYA_CUDA_LIBRARY`, `LAYA_CUDA_DEVICE`, `LAYA_CHECKPOINT` and `LAYA_WEIGHT_ORACLE`.
It uploads all 205 used tensors and compares readback hashes with the existing Torch
oracle; it does not test model outputs or latency.

## Inference workspace

`Workspace::new(&cuda, batch, sequence)` allocates fixed scratch buffers for one
shape. Batch must be 1, 2, 4, 8 or 16; sequence must be a multiple of 16 in 16..=512.
Invalid shapes fail before any allocation; a failed allocation releases the partial
workspace. Contents are uninitialized and must be written before use.

`buffers()` borrows the named buffers without allowing allocations to be replaced.
The workspace outlives the caller's `Cuda` handle. `bytes()` reports scratch
allocations only, excluding resident weights and CUDA overhead.

Let `B` be batch, `L` sequence, `D=1024`, and `M=MAX_MARKERS=2048`. Layouts are:

| Buffers | Shape and dtype |
| --- | --- |
| ids / lengths / types | `[B,L]` int64 / `[B]` int32 / `[B]` int64 |
| residual / hidden / attention | `[B,L,D]` FP32 / BF16 / BF16 |
| qkv / gated / feed_forward | `[B,L,3D]` / `[B,L,2624]` / `[B,L,4096]`, BF16 |
| indices / offsets | `[M]` / `[B+1]`, int32 |
| markers / scored / logits | `[M,D]` / `[M,D]` / `[M]`, BF16 |
| features / action_hidden / actions | `[B,1028]` / `[B,256]` / `[B,2]`, BF16 |

At `(B,L)=(1,512)`, allocations total 22,628,896 bytes; at `(16,512)`,
236,048,836 bytes. The caller must enforce at most `MAX_MARKERS` scored positions.
No Graph cache, kernel launch, cuBLAS workspace or inference is included.

The ignored `real_gpu_workspace_capacity_and_reuse` test requires
`LAYA_CUDA_LIBRARY` and `LAYA_CUDA_DEVICE`. It writes and reads all 17 buffers twice
at three shapes, including both capacity bounds, using deterministic byte patterns.
This tests allocation and transfer, not model numerics or latency.

## Eager encoder

`Encoder::load(&cuda, checkpoint, bundle)` loads the existing trusted native Laya
bundle and resident weights. `run(ids, lengths, types, &workspace)` executes all
28 encoder layers and both decision transformer layers, synchronizes, and leaves
FP32 hidden states in `workspace.buffers().residual`. Padded rows have length zero.
IDs must be within the vocabulary; lengths and types are checked before upload.

This uses the existing original RoPE entry point and dynamic full/local attention.
No Scorer, output decoding, HTTP service, Graph capture or cache is included.
The resource library supplied to `Cuda::load` remains separate from the operator
bundle; CPU builds need neither library. The operator bundle currently targets
Hopper `sm_90a`. `Encoder::load` is unsafe because callers must trust the native
code and provide a compatible GPU; hashes bind artifacts, not code trust.

Reuse the [existing CUDA build entry](https://github.com/linear3735/system1-omni/blob/5ff41a5/src/backends/cuda/build.sh)
and its model source/export tools. Export rotary tables with that version's
`tools/export_tables.py`. Keep `liblaya_cuda.so`, `build-manifest.json`,
`tables.json` and the four rotary table files in one bundle directory. These build
tools are a separate dependency, not duplicated by this encoder change. Startup
checks checkpoint and bundle hashes before loading native code.

For GPU validation, supply real request fixtures to
`recipe/laya/native/export_encoder.py CHECKPOINT REQUESTS OUTPUT`. It uses Laya
0.3.20 and PyTorch on CUDA as the reference, independently of the Rust runtime.
Set `LAYA_CUDA_LIBRARY`, `LAYA_CHECKPOINT`, `LAYA_KERNEL_BUNDLE` and
`LAYA_ENCODER_ORACLE`, then run:

```sh
cargo test --release --locked -p omni-laya --lib real_encoder_matches_official_hidden_states -- --ignored --nocapture
```

The test checks selected encoder intermediates, both head layers, final hidden
states, shape changes and repeated workspace reuse. It compares valid tokens;
empty-key padding differs intentionally from the original attention. Candidate
padding must still be finite. This is numerical parity, not model quality or a
performance benchmark. The intermediate capture hooks compile only in tests.
