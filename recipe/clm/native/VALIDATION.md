# CLM CUDA validation

`omni-clm` owns everything after the encoder: a frozen Qwen3-8B runs as its own process
behind an `/v1/embeddings` endpoint, and the engine owns the two projection heads, the
cosine score, the temperature and the typed answer. Agreement with the reference is
therefore two separate questions — whether the engine's own arithmetic matches, and whether
the whole path still agrees once a real encoder is in front of it — and they are measured
separately below, because one number cannot answer both.

Both phases ran on one RTX 4090 on 2026-10-06. Every number below is the output of
`compare_with_reference.py`, which sends the same requests to `clm-run` and to CLM's own
`Engine` and `Schema` and compares the answers.

## The environment

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24 GB, compute capability 8.9 |
| Driver | 595.71.05 |
| CUDA | 13.0, V13.0.88 |
| Python | 3.12.3 |
| torch | 2.12.1+cu130 |
| transformers | 5.17.0, the encoder |
| contrastive-lm | 0.1.0, the reference |
| Rust | 1.98.1 |
| Engine commit | `a2019f9` |
| `clm-run` | sha256 `791859b51047edd79ad6f380bba9a78ea1fdddb5c4a8ad797472ac541251dc92` |

## Phase A: the engine alone

`recipe/clm/stub_embedder.py` derives its vectors from the text alone, and both sides ask
it, so the same text reaches both as the same vector. The encoder's numerical difference is
out of the comparison; what remains is parsing, the text the heads see, the projections, the
cosine, the temperature and the answer shape — and, because a difference in the text the two
sides send would produce a different vector, the text is in the comparison too.

| case | kind | agreement |
| --- | --- | --- |
| `choice_two` | choice | `max\|dp\|=3.12e-07`, both pick `billing` |
| `choice_five` | choice | `max\|dp\|=4.49e-06`, both pick `c` |
| `score_three` | score | `max\|dp\|=1.13e-06`, `1.132559` vs `1.132558` |
| `noul_stmt` | noul | `0.271456` vs `0.271451` |
| `noul_numbers` | noul | `0.225941` vs `0.225943` |

Worst case **4.49e-06**. Nothing here is passed a tolerance to hide behind: the answers are
compared at the 0.025 the script uses for the real encoder, and they land more than three
orders of magnitude inside it.

## Phase B: end to end, on Qwen3-8B

`recipe/clm/native/transformers_encoder.py` serves Qwen3-8B — the model the deployment
encodes with — behind the same endpoint, so vLLM is not needed to check the client.

| case | kind | agreement |
| --- | --- | --- |
| `choice_two` | choice | `max\|dp\|=1.72e-05`, both pick `billing` |
| `choice_five` | choice | `max\|dp\|=1.97e-02`, both pick `a` |
| `score_three` | score | `max\|dp\|=1.08e-02`, `0.385166` vs `0.403241` |
| `noul_stmt` | noul | `0.631038` vs `0.646805` |
| `noul_numbers` | noul | `0.323871` vs `0.321280` |

Worst case **1.97e-02**, against the script's tolerance of 0.025, and all five cases agree.

## Why the two numbers differ, and why the tolerance is 0.025

CLM scores with `softmax(exp(logit_scale) * cos(...) / temperature)`, and the published
checkpoint's `logit_scale` is 4.6132, whose exponential the engine caps at 100. A difference
of `d` in a cosine therefore becomes `100 d` in the logit, and two forward passes that
disagree by ~1e-4 in the cosine — which is what two encoder implementations produce for
Qwen3-8B — disagree by ~1e-2 in the probabilities.

Phase A is what makes that attribution a measurement rather than a story: with the encoder
taken out, the engine agrees to 4.49e-06, so the ~1e-2 in phase B is the encoder and not the
engine. A deployment serving the same Qwen3-8B weights would be on the other side of that
difference, which is a property of how this model scores, not a defect on either side.

## Artifacts

| file | sha256 |
| --- | --- |
| `CLM_v0.1-8B.pt` | `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5` |
| `clm-export/oracle.json` | `7d058de4a691b6d4949aa51c53c7ec95a136ac4c14290580e2d81ba8f7192c16` |
| Qwen3-8B `config.json` | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |

`clm-export/model.safetensors` is deliberately not pinned. The safetensors serializer
carries `__metadata__` through a hash map, so its key order — and with it the file hash —
differs from run to run: four exports of this same checkpoint gave four file hashes while
their tensor entries were byte-identical and every one of them wrote the same `oracle.json`.
`oracle.json` holds the FP32/FP16/BF16 hash of every tensor and is stable, byte-identical
across two machines, so that is what pins the conversion.

## Reproduction

Both phases take the same comparison; only the endpoint changes.

```sh
# the checkpoint, once
python recipe/clm/native/export_weights.py CLM_v0.1-8B.pt clm-export

# Phase A, no GPU and no 8B encoder
python recipe/clm/stub_embedder.py --port 8090 &

# Phase B, instead of the stub, on a GPU
python recipe/clm/native/transformers_encoder.py --model /path/to/Qwen3-8B --port 8090 &

python recipe/clm/native/compare_with_reference.py \
  --checkpoint clm-export --bin target/release/clm-run \
  --emb-url http://127.0.0.1:8090/v1/embeddings \
  --pt CLM_v0.1-8B.pt --temperature 1.0
```

## What this does not establish

- **Decision quality.** Five fixed requests show the port is faithful; they are not a
  labelled set and say nothing about how often CLM is right.
- **vLLM.** The encoder here is Transformers. A deployment's pooling server is a different
  implementation and sits on the other side of the ~1e-2 above; nothing here measures it.
- **Latency or throughput.** Nothing was timed, and there is no candidate-vector cache to
  exercise: `Engine::decide` embeds every text on every call.
- **Other hardware.** One RTX 4090, compute capability 8.9. The engine is Rust and has no
  device-specific code; the CUDA part of this path is `transformers_encoder.py`.
