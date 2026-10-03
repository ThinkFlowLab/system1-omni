#!/usr/bin/env python3
"""Deterministic /v1/embeddings stand-in for the Qwen3-8B pooling server.

Why this exists: `clm-serve` is an HTTP client of an OpenAI-compatible embeddings
endpoint (`src/clm/embedder.py`), so the engine, the packing, the scoring and the
HTTP API can all be exercised without a GPU or the 8B encoder. The vectors here are
not meaningful - they are a hash-derived direction on the unit sphere - so this
verifies the *plumbing* (request shape, ordering, cache, the three question types,
the serving contract), never the model's decisions.

    python stub_embedder.py --port 8090 --dim 4096
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 4096


def vector(text: str, dim: int) -> list[float]:
    """A stable unit vector per text: two hash digests expanded by a counter."""
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        for i in range(0, len(digest) - 3, 4):
            if len(out) == dim:
                break
            word = int.from_bytes(digest[i:i + 4], "big")
            out.append(word / 2**31 - 1.0)
        counter += 1
    norm = sum(v * v for v in out) ** 0.5 or 1.0
    return [v / norm for v in out]


class Handler(BaseHTTPRequestHandler):
    dim = DIM
    calls = 0

    def log_message(self, *args):  # keep the run quiet
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
            self._json(200, {"object": "list", "data": [{"id": "qwen3-8b", "object": "model"}]})
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
        Handler.calls += len(texts)
        data = []
        tokens = 0
        for index, text in enumerate(texts):
            vec = vector(text, Handler.dim)
            raw = struct.pack(f"<{len(vec)}f", *vec)
            data.append({"object": "embedding", "index": index,
                         "embedding": base64.b64encode(raw).decode()})
            tokens += max(1, len(text) // 4)
        self._json(200, {"object": "list", "data": data, "model": body.get("model", "qwen3-8b"),
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--dim", type=int, default=DIM)
    args = parser.parse_args()
    Handler.dim = args.dim
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"stub embedder on http://127.0.0.1:{args.port} (dim {args.dim})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
