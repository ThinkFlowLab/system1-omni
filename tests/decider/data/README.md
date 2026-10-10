# Pinned CPU fixtures

`cpu.json` retains 12 request/token/row/usage fixtures and all 255 label IDs from
Mapika/decider `50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f`, using Decider-2B v11
`533964dae8be954c5b5e19fa4948e48408094c1e`. The tokenizer SHA-256 is checked by
`Processor::load`; the tests require only tokenizer and configuration files.
The large truncation fixture is replaced with generated assertions in `contract.rs`.

`responses.json` retains 12 pinned Torch 2.14.0 CPU assemblies, including isolated
Score, 255-label uniform Choice, and zero-fit Score. Inputs are synthetic, with no
private text. Apache-2.0 reference attribution is retained in the native crate.

Run from the repository root:

```sh
DECIDER_MODEL=/path/to/decider cargo test --locked -p omni-decider-native \
  --test contract -- --ignored
```

`verify_reference.py` independently reconstructs prompts with pinned upstream
`prompt_fast`, `systemone` and `temperature`, then compares actual BF16 CUDA
checkpoint outputs and complete native responses. It saves its protocol before
execution and retains raw rows/logits/responses outside the repository fixtures.

## Prefix workloads

`prefix-workloads.json` is the frozen request-local prefix comparison corpus.
It is consumed by `tests/decider/verify_reference.py --cases
 tests/decider/data/prefix-workloads.json` after building the registered
`decider-run` example. It supplements the default 18-case reference corpus and
requires the pinned model, reference package and real CUDA library. Generated
outputs belong outside this data directory.
