"""HTTP worker for Cua-S1 4B 0.2 (`text` adapter): `GET /health` and `POST /v1/systemone`.

PYTHONPATH=src python -m frontend.cua_s1_text --base <dir> --adapter <dir>/text
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# FastAPI reads the handler annotations at runtime, so `Request` must be a
# module-level name while `from __future__ import annotations` is in effect.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from models.cua_s1.text.contract import (
    MODEL_ID,
    RequestError,
    answer,
    map_request,
    parse_body,
)

MAX_BODY_BYTES = 4 << 20
MAX_PROMPT_TOKENS = 16384
WARMUP = (
    b'{"model": "cua-s1-4b-0.2", "state": "Dialog: Update installed.", "questions": {"q":'
    b' {"type": "choice", "instructions": "Close it.", "criteria": {"ok": "OK", "wait": "Wait"}}}}'
)


def build_app(model: Any) -> FastAPI:
    app = FastAPI()
    pool = ThreadPoolExecutor(max_workers=1)  # one forward pass at a time

    def decide(raw: bytes) -> dict[str, Any]:
        state, questions = map_request(parse_body(raw))
        encoded = [model.encode(state, q) for q in questions]
        tokens = [int(x["input_ids"].shape[1]) for x in encoded]
        for q, n in zip(questions, tokens):  # before any forward pass
            if n > MAX_PROMPT_TOKENS:
                raise RequestError(
                    f"question {q.name!r}: {n} prompt tokens, over {MAX_PROMPT_TOKENS}",
                    413,
                )
        answers = {
            q.name: answer(q, model.score(x, len(q.keys)))
            for q, x in zip(questions, encoded)
        }
        return {
            "model": MODEL_ID,
            "answers": answers,
            "usage": {"input_tokens": sum(tokens), "output_tokens": 0},
        }

    @app.get("/health")
    def health():
        return {"status": "ready", "model": MODEL_ID}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > MAX_BODY_BYTES:
                return JSONResponse({"detail": "request body too large"}, 413)
        try:
            # JSONResponse, not FastAPI's encoder, which drops keys starting with "_sa".
            return JSONResponse(
                await asyncio.get_running_loop().run_in_executor(pool, decide, raw)
            )
        except RequestError as error:
            return JSONResponse({"detail": str(error)}, error.status)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return JSONResponse({"detail": "inference failed"}, 500)

    # One decision on the worker thread before listening, so the first request does not
    # pay for first-call setup there.
    app.state.warmup = lambda: pool.submit(decide, WARMUP).result()
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="local Qwen/Qwen3.5-4B directory")
    parser.add_argument(
        "--adapter", required=True, help="local text/ adapter directory"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from models.cua_s1.text.model import TextModel

    app = build_app(TextModel(args.base, args.adapter, args.device, args.dtype))
    app.state.warmup()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
