#!/usr/bin/env python3
"""A `/v1/embeddings` endpoint backed by transformers, for verifying omni-clm.

`vllm serve --runner pooling` is the deployment shape, but vLLM is a large install and
this is only needed to produce vectors for a parity check. Transformers loads the same
Qwen3-8B checkpoint and pools the last token, which is what CLM's heads were trained
against — `serve_qwen3_8b.sh` in the CLM repository uses vLLM with `--runner pooling`
precisely because it does the same thing.

    python recipe/clm/native/transformers_encoder.py --model /path/to/Qwen3-8B --port 8090

Not for production: one request at a time, no batching across callers.
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = None
TOKENIZER = None
LOCK = threading.Lock()


def embed(texts: list[str]) -> list[list[float]]:
    import torch

    with LOCK:
        batch = TOKENIZER(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        batch = {k: v.to(MODEL.device) for k, v in batch.items()}
        with torch.no_grad():
            out = MODEL(**batch)
        # Last-token pooling, as CLM's encoder does and its heads were trained on.
        mask = batch["attention_mask"]
        last = mask.sum(dim=1) - 1
        rows = torch.arange(mask.shape[0], device=mask.device)
        hidden = out.last_hidden_state[rows, last]
    return hidden.float().cpu().tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": ARGS.model_name, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/v1/embeddings"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        texts = body.get("input") or []
        if isinstance(texts, str):
            texts = [texts]
        try:
            vectors = embed(texts)
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        data = []
        tokens = 0
        for index, vec in enumerate(vectors):
            raw = struct.pack(f"<{len(vec)}f", *vec)
            data.append({
                "object": "embedding",
                "index": index,
                "embedding": base64.b64encode(raw).decode(),
            })
            tokens += len(TOKENIZER.tokenize(texts[index]))
        self._json(200, {
            "object": "list",
            "data": data,
            "model": body.get("model", ARGS.model_name),
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        })


def main() -> None:
    global MODEL, TOKENIZER, ARGS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="path to Qwen3-8B")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model-name", default="qwen3-8b")
    parser.add_argument("--dtype", default="bfloat16")
    ARGS = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"loading {ARGS.model} ({ARGS.dtype})", flush=True)
    TOKENIZER = AutoTokenizer.from_pretrained(ARGS.model)
    MODEL = AutoModel.from_pretrained(ARGS.model, dtype=getattr(torch, ARGS.dtype))
    MODEL = MODEL.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    print(f"ready on {MODEL.device}, hidden {MODEL.config.hidden_size}", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"listening on http://127.0.0.1:{ARGS.port}/v1/embeddings", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
