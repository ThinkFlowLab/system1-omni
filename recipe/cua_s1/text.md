# Cua-S1 4B 0.2 text worker

This recipe runs the Cua-S1 4B 0.2 `text` adapter through Transformers and PEFT behind the Rust frontend. It is the correctness reference for native execution. The model is in [`src/models/cua_s1/text/`](../../src/models/cua_s1/text/), the HTTP worker in [`src/frontend/cua_s1_text.py`](../../src/frontend/cua_s1_text.py), and [`src/models/cua_s1/README.md`](../../src/models/cua_s1/README.md) documents the contract. Only `choice` questions are supported.

Run the commands from the repository root, on Linux with an NVIDIA GPU and Python 3.12. The pinned versions match the upstream reference environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r recipe/cua_s1/requirements-text.txt
.venv/bin/hf download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir weights/Qwen3.5-4B
.venv/bin/hf download cua-ai/cua-s1-4b-0.2 \
  --revision 16818868b0cc7813808aae4e87b417657046ab79 --local-dir weights/cua-s1-4b-0.2
```

Upstream's `libs/cua-s1/ci/fetch_pinned_weights.py --dest weights --verify-only` (in [trycua/cua](https://github.com/trycua/cua) at `0e75660ce4c2edda519e0c795fa3ad98abf4e76f`) checks every downloaded file against upstream's lock.

Start the worker, which runs one warmup decision before it listens, then the frontend:

```sh
PYTHONPATH=src .venv/bin/python -m frontend.cua_s1_text \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text --port 8000
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 ./target/release/omni-jev
```

Requests run one at a time. Bodies over 4 MiB, more than 64 questions, or a prompt over 16,384 tokens get `413`. The worker computes logits for every position, as upstream does, so memory grows with prompt length: the 15,446-token test input peaked at about 21.3 GiB in bfloat16.

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"cua-s1-4b-0.2","state":"Dialog: Delete 3 files permanently? Buttons: Delete, Cancel","questions":{"pick":{"type":"choice","instructions":"Keep the files.","criteria":{"delete":"Click Delete","cancel":"Click Cancel"}}}}'
```

On an RTX 6000 Ada in bfloat16, the response is:

```json
{"model":"cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:text","answers":{"pick":{"type":"choice","choice":"cancel","probabilities":{"delete":0.0024726232513785362,"cancel":0.9975274205207825},"confidence":0.9750249565060322}},"usage":{"input_tokens":153,"output_tokens":0}}
```

The tests need neither weights nor a GPU; with `CUA_S1_BASE=weights/Qwen3.5-4B` they also check the tokenizer:

```sh
.venv/bin/python -m pip install pytest httpx
PYTHONPATH=src .venv/bin/python -m pytest tests/cua_s1
```
