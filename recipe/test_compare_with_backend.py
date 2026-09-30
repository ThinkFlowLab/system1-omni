#!/usr/bin/env python3
"""Regression tests for compare_with_backend.py.

Stands up two throwaway HTTP servers -- one acting as the worker, the other as the
frontend -- and checks that the comparison passes when only ``usage`` differs but still
fails when an answer or a status differs. Runs without a model, a GPU or the Rust binary.

    python recipe/test_compare_with_backend.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "compare_with_backend.py"

ANSWERS = {"pick": {"type": "noul", "noul": 0.75}}
HEALTH = {"ok": True}


class Server(ThreadingHTTPServer):
    """A worker whose response the test controls, plus a call counter."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.calls = 0
        self.answers = ANSWERS
        self.usage: dict = {"input_tokens": 0}
        # When set, every other call reports this instead -- a cache that goes warm.
        self.alternate_usage: dict | None = None
        self.status = 200
        # The body sent with a non-200 status; normally an empty object.
        self.error: dict = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output readable
        pass

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, json.dumps(HEALTH).encode())

    def do_POST(self) -> None:  # noqa: N802
        server: Server = self.server  # type: ignore[assignment]
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        server.calls += 1
        if server.status != 200:
            self._send(server.status, json.dumps(server.error).encode())
            return
        usage = dict(server.usage)
        if server.alternate_usage is not None and server.calls % 2 == 0:
            usage.update(server.alternate_usage)
        self._send(200, json.dumps({"model": "m", "answers": server.answers, "usage": usage}).encode())


def run(backend: str, frontend: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--backend", backend, "--frontend", frontend, "--model", "m"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


def serve(server: Server) -> None:
    threading.Thread(target=server.serve_forever, daemon=True).start()


def reset(*servers: Server) -> None:
    for server in servers:
        server.calls = 0
        server.answers = ANSWERS
        server.usage = {"input_tokens": 0}
        server.alternate_usage = None
        server.status = 200
        server.error = {}


def main() -> None:
    worker, frontend = Server(), Server()
    serve(worker)
    serve(frontend)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    # 1. Identical responses: the strict comparison passes.
    reset(worker, frontend)
    code, out = run(worker.url, frontend.url)
    check("identical responses pass", code == 0 and "FAIL" not in out, out)

    # 2. Only usage differs -- the cache-aware case. Must pass, and say so.
    reset(worker, frontend)
    worker.alternate_usage = {"input_tokens": 26}
    code, out = run(worker.url, frontend.url)
    check("usage-only difference passes and is reported",
          code == 0 and "FAIL" not in out and "input_tokens" in out, out)

    # 3. An answer differs: must still fail.
    reset(worker, frontend)
    frontend.answers = {"pick": {"type": "noul", "noul": 0.1}}
    code, out = run(worker.url, frontend.url)
    check("a differing answer fails", code == 1 and "FAIL" in out, out)

    # 4. A status difference must fail even when the answers match.
    reset(worker, frontend)
    frontend.status = 503
    code, out = run(worker.url, frontend.url)
    check("a status difference fails", code == 1 and "FAIL" in out, out)

    # 5. A backend that is not 200 must fail regardless of the frontend.
    reset(worker, frontend)
    worker.status = 500
    code, out = run(worker.url, frontend.url)
    check("a non-200 backend fails", code == 1 and "FAIL" in out, out)

    # 6. Matching 500s are not a pass, even when the error body carries equal answers.
    reset(worker, frontend)
    worker.status = 500
    frontend.status = 500
    worker.error = {"answers": {}, "detail": "failed"}
    frontend.error = {"answers": {}, "detail": "failed"}
    code, out = run(worker.url, frontend.url)
    check("equal answers on matching errors fail", code == 1 and "FAIL" in out, out)

    worker.shutdown()
    frontend.shutdown()

    failed = 0
    for name, ok, detail in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failed += 1
            print("       " + detail.strip().replace("\n", "\n       ")[:400])
    print("compare_with_backend: ok" if not failed else f"compare_with_backend: {failed} FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
