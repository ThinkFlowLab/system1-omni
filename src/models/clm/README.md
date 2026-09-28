# CLM model engine

CLM is the second model the project tracks ([#9](https://github.com/ThinkFlowLab/system1-omni/issues/9)), after LAYA. It decides differently in a way LAYA does not cover: **the engine does not compute embeddings.** A frozen `Qwen/Qwen3-8B` encoder runs as its own process behind an OpenAI-compatible `/v1/embeddings` endpoint, and the engine owns everything after it — two projection heads, the cosine score, and the typed answer.

That split is the point of implementing it second. LAYA's engine owns one forward pass; this one owns a client to someone else's server, plus a candidate-vector cache that persists across requests.

## What is here

`omni-clm` reads a converted checkpoint and computes decisions. It does not call an embeddings endpoint and does not serve HTTP yet; those belong with the runtime that owns the request path.

| module | responsibility |
| --- | --- |
| `config` | the head geometry, read from the safetensors metadata |
| `weights` | tensor inventory, shape checks, FP32 loading |
| `scoring` | projection, cosine score, softmax, and the three answer types |

## The checkpoint is converted first

A CLM checkpoint is a `torch.save` dict, so it is a pickle and no non-Python reader can open it. `recipe/clm/native/export_weights.py` writes the tensors to safetensors with the head name as a prefix and keeps `cfg`, `hidden_size`, `projection_dim` and `logit_scale` in the metadata.

```sh
python recipe/clm/native/export_weights.py CLM_v0.1-8B.pt OUT_DIR
```

`CLM_v0.1-8B.pt` is the published checkpoint from `Contrastive-LM/CLM-v0.1-8B`; it is 75 MB and holds the two heads, not the 8B encoder. The export is 16 tensors, 18.9 M parameters.

## The decision, in one pass

A decision is `softmax(exp(logit_scale) * cos(state_head(s), action_head(c)) / temperature)` over a question's candidates. Both heads are `inp → [LayerNorm →] hidden → out` with GELU, and both projections are L2-normalised before the dot product; the published checkpoint sets `layernorm: true` and `residual: false` with one hidden block.

The three question types differ only after the distribution exists, which is why they share one scoring path:

| type | answer |
| --- | --- |
| `choice` | the argmax key, with `confidence` and the full distribution |
| `noul` | the `true` entry of a two-candidate distribution |
| `score` | the expected level index, `sum(i * p_i)`, with `confidence` |

`confidence` is the top probability minus the mean of the rest, clamped to `[0, 1]`, and `1.0` for a single candidate — the TypeSafe-style definition `schema.py` uses.

## CPU checks

The default tests need no checkpoint:

```sh
cargo test -p omni-clm
```

To check the full checkpoint and the decision arithmetic, export it first and point `CLM_EXPORT` at the directory:

```sh
python recipe/clm/native/export_weights.py /path/to/CLM_v0.1-8B.pt /tmp/clm-export
python recipe/clm/native/head_oracle.py /tmp/clm-export /tmp/clm-export/head-oracle.json
CLM_EXPORT=/tmp/clm-export cargo test -p omni-clm -- --ignored
```

Three tests run there: every tensor's FP32 conversion hash against the export oracle, the inventory against the head configuration, and **five decisions checked against an independent NumPy implementation of the same arithmetic** (`head_oracle.py`) on synthesised embeddings, so the two sides need no encoder to disagree. The embeddings are hash-derived and carry no meaning as model output — the check is that two implementations of the same maths agree.

The default CI job skips these because it does not download the checkpoint.
