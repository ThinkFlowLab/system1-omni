# CLM behind the frontend

Runs [CLM](https://github.com/Contrastive-LM/CLM)'s own `clm-serve` behind `omni-jev`, and checks that the
frontend returns what the engine returned. CLM is the second model in #9: a frozen Qwen3-8B encoder behind an
OpenAI-compatible `/v1/embeddings` endpoint, plus two projection heads and a cosine score.

**No GPU and no 8B encoder are needed to run this.** `clm-serve` is an HTTP client of the embeddings endpoint
(`src/clm/embedder.py`), so `stub_embedder.py` can stand in for the encoder. What that exercises is the
*plumbing* — request shape, question packing, the three answer types, the serving contract, and the frontend in
front of it. It cannot tell you anything about CLM's decisions, because the vectors are not from Qwen.

## Run it

Three terminals, all CPU:

```bash
# 1. a stand-in for the Qwen3-8B pooling server
python recipe/clm/stub_embedder.py --port 8090

# 2. CLM's own server, pointed at it (the checkpoint is 75 MB: the two heads, not the encoder)
CLM_CKPT_DIR=/tmp/clm-ckpt clm-serve --port 8091 \
  --emb-url http://127.0.0.1:8090/v1/embeddings --emb-model qwen3-8b

# 3. the frontend from #2, pointed at CLM
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8091 cargo run -p omni-jev --release
```

```bash
python recipe/compare_with_backend.py \
  --backend http://127.0.0.1:8091 --frontend http://127.0.0.1:8080 --model clm-latest
```

Installing CLM without the GPU stack, since `vllm` is a hard dependency of the package but is only needed for
the encoder process:

```bash
pip install "numpy>=1.24" requests "fastapi>=0.100" "uvicorn>=0.23" torch
pip install --no-deps "contrastive-lm @ git+https://github.com/Contrastive-LM/CLM.git"
```

## What passes

The contract lines up with no adapter: status, content type and all three answer types come back through the
frontend unchanged, including CLM's `X-CLM-Latency-Ms`.

| question | answer |
|---|---|
| `choice` | `{"type":"choice","choice":"billing","confidence":…,"probabilities":{…}}` |
| `score` | `{"type":"score","score":0.750,"confidence":…,"legend":{"0":…},"probabilities":{…}}` |
| `noul` | `{"type":"noul","noul":0.9995}` |

## What the comparison had to learn

`compare_with_backend.py` compared the whole response byte-for-byte, which holds for LAYA because its body is a
pure function of the request. It does not hold for an engine that reuses encoder state across requests:

```
same state, three times:   noul=0.977197  usage.input_tokens=0
a state not seen before:   noul=0.280477  usage.input_tokens=26
that same state again:     noul=0.280477  usage.input_tokens=0
```

The decision is deterministic to six decimals; `input_tokens` counts only encoder cache misses. CLM's whole
point is that candidate vectors are reusable across requests, so the field is moved by the feature that makes
it interesting. Comparing the full body reports FAIL on a correct response, and whether it does depends on
which call happened to warm the cache — so the same run can pass or fail on ordering.

The tool now compares status, content type and the `answers` subtree, and prints the `usage` difference instead
of asserting on it. Strict equality is still the first test, so a backend whose body really is a pure function
is unaffected:

```
PASS department: status 200 -> 200  (usage {'billing_units': 1, 'input_tokens': 33, 'output_tokens': 0} -> {'billing_units': 1, 'input_tokens': 0, 'output_tokens': 0})
```

## Open contract question

Which response fields are allowed to differ between two otherwise identical requests? `billing_units` looks
stable; `input_tokens` does not. If the project wants the strong form — the whole body identical — then
`input_tokens` has to mean "tokens the request required" rather than "tokens this call paid for", which is a
decision for the engine, not for the frontend.

## With a real encoder

Replace step 1 with the upstream script and the answers become meaningful:

```bash
GPU=0 PORT=8090 UTIL=0.35 ./serve_qwen3_8b.sh     # from the CLM checkout; needs CUDA
```

Everything downstream is unchanged, which is the property this recipe is meant to demonstrate.
