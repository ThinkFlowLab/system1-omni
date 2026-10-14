"""Serve the upstream CLM API with request-to-request vector caches disabled.

Use a prepared CLM checkout on PYTHONPATH and a separately reserved real encoder.
The model, head arithmetic and API implementation remain upstream's.
"""

import argparse

import uvicorn
from clm.embedder import Embedder
from clm.engine import Engine
from clm.server import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--emb-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    embedder = Embedder(
        url=args.emb_url, model="qwen3-8b", max_tokens=2048, cache_size=0
    )
    engine = Engine(
        embedder=embedder, checkpoint=args.checkpoint, device="cuda", action_cache=0
    )
    uvicorn.run(create_app(engine, ui=False), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
