"""Python reference worker for Valen-Preview-0923.

The Rust frontend remains the public transport layer. This worker owns only
HTTP lifecycle and request orchestration; preprocessing lives in
models.valen.preprocess.ValenProcessor and model execution lives in
models.valen.engine.ValenExecutor.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from models.valen.engine import REFERENCE_REVISION, ExecutorOutput, ValenExecutor
from models.valen.postprocess import answer, build_response
from models.valen.preprocess import ValenProcessor, prepare_request
from models.valen.protocol import (
    MAX_BODY,
    MODEL_NAME,
    RequestError,
    parse_body,
    parse_request,
)

LOG = logging.getLogger(__name__)


def decide(raw: bytes, processor: ValenProcessor, executor: ValenExecutor) -> dict:
    """Prepare, compile, execute, and finish one complete request."""

    request = parse_request(parse_body(raw))
    with TemporaryDirectory(prefix="system1-valen-") as directory:
        prepared = prepare_request(request, Path(directory))
        compiled = processor.compile(prepared)
        output = executor.execute(compiled)
        if not isinstance(output, ExecutorOutput):
            raise RuntimeError("Valen executor returned an invalid output object")
        if len(output.logits) != len(compiled.response_context.questions):
            raise RuntimeError("Valen executor returned an unexpected answer count")
        answers = {
            question.name: answer(question, logits)
            for question, logits in zip(compiled.response_context.questions, output.logits)
        }
        return build_response(
            compiled.response_context,
            answers,
            output.logical_tokens,
            output.compute_tokens,
        )


def _warmup_request():
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(stream, "PNG")
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return parse_request(
        {
            "model": MODEL_NAME,
            "state": {"image": f"data:image/png;base64,{encoded}"},
            "questions": {
                "_warmup": {
                    "type": "choice",
                    "instructions": "Choose the only candidate.",
                    "criteria": {"ok": "OK"},
                }
            },
        }
    )


def warmup(processor: ValenProcessor, executor: ValenExecutor) -> None:
    request = _warmup_request()
    with TemporaryDirectory(prefix="system1-valen-warmup-") as directory:
        prepared = prepare_request(request, Path(directory))
        executor.warmup(processor.compile(prepared))


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, processor: ValenProcessor, executor: ValenExecutor):
        self.processor = processor
        self.executor = executor
        self.inference_lock = threading.Lock()
        self._executor_closed = False
        super().__init__(address, Handler)

    def server_close(self):
        super().server_close()
        if not self._executor_closed:
            self.executor.close()
            self._executor_closed = True


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        # Do not log paths, images, prompts, or arbitrary request headers.
        pass

    def send_json(self, status: int, value: dict):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ready", "model": MODEL_NAME, "modality": "multimodal"})
        else:
            self.send_json(404, {"detail": "unknown route"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self.send_json(404, {"detail": "unknown route"})
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_json(411, {"detail": "Content-Length is required; chunked requests are unsupported"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            self.send_json(411, {"detail": "one Content-Length is required"})
            return
        try:
            length = int(lengths[0])
        except ValueError:
            self.send_json(400, {"detail": "invalid Content-Length"})
            return
        if length < 0 or length > MAX_BODY:
            self.send_json(413, {"detail": "request exceeds body limit"})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"detail": "Content-Type must be application/json"})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.send_json(503, {"detail": "worker busy"})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                self.send_json(400, {"detail": "incomplete body"})
                return
            self.send_json(200, decide(raw, self.server.processor, self.server.executor))
        except RequestError as exc:
            self.send_json(exc.status, {"detail": str(exc)})
        except (TimeoutError, socket.timeout):
            self.send_json(408, {"detail": "request body timed out"})
        except Exception:
            LOG.exception("Valen inference failed")
            self.send_json(500, {"detail": "inference failed"})
        finally:
            self.server.inference_lock.release()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valen-source", required=True, help="pinned Valen source checkout")
    parser.add_argument("--checkpoint", required=True, help="Valen-Preview-0923 directory")
    parser.add_argument("--base", required=True, help="Qwen3.5-2B base directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def _verify_source_revision(source: Path) -> None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("--valen-source must be a git checkout") from exc
    if result.stdout.strip() != REFERENCE_REVISION:
        raise SystemExit(
            f"--valen-source must be pinned to {REFERENCE_REVISION}"
        )


def main(argv=None):
    args = parse_args(argv)
    source = Path(args.valen_source).resolve()
    if not (source / "valen").is_dir():
        raise SystemExit("--valen-source must contain the pinned valen package")
    _verify_source_revision(source)
    sys.path.insert(0, str(source))
    config = json.loads(
        (Path(args.checkpoint).resolve() / "config.json").read_text(encoding="utf-8")
    )
    logging.basicConfig(level=logging.INFO)
    processor = ValenProcessor(
        args.base,
        int(config["max_length"]),
        config.get("media_kwargs"),
    )
    executor = ValenExecutor(args.checkpoint, args.base, args.device, args.dtype)
    server = None
    try:
        warmup(processor, executor)
        server = WorkerServer((args.host, args.port), processor, executor)
        LOG.info("Valen worker ready on %s:%s", args.host, args.port)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        else:
            executor.close()


if __name__ == "__main__":
    main()
