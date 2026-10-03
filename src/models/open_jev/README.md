# Open-Jev-27B-v1.1

The [native Rust/CUDA worker](../../../recipe/open_jev/native.md) uses the
pinned Qwen3.8-27B backbone, merged LoRA adapter, trained scalar decision head,
and saved calibration temperature. It supports choice, ordinal score, and yes/no
text decisions through the existing Rust frontend.

The request compiler and response formulas follow
[Open-Jev @ 3308a15](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16).
The CUDA prefill implementation is shared with Cua-S1 under
[`../qwen3_5/native/`](../qwen3_5/native/). See the recipe for preparation,
numerical limitations, validation and optimization scope.

The Rust contract is adapted from Open-Jev's MIT-licensed code; its copyright
and license are retained in [`native/LICENSE.open-jev`](native/LICENSE.open-jev).
No model weights are distributed here.
