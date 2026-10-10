# Decider native contract attribution

The request rendering, row planning, token construction and answer formulas in
`src/contract.rs` and `src/processing.rs` adapt Mapika/decider's
`decider/systemone.py`, `decider/prompt.py`, `decider/prompt_fast.py` and
`decider/temperature.py`, revision
`50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f` (decider-ai 1.8.1).
Copyright the Mapika/decider contributors. Licensed under Apache-2.0;
see [the retained license](LICENSE.decider).

The adaptation uses Rust CPU buffers, the frozen Decider-2B v11 tokenizer and
configuration, explicit native input/admission restrictions, independent rows
and isolated Score levels. Native execution reuses the repository Qwen3.5 CUDA
backbone and the reference BF16 selected tied-embedding projection contract. The model-private JSON renderer follows this repository's shared JSON
helper's Python notation conventions, with stricter integer decoding to prevent
arbitrary_precision feature unification from changing state values.
