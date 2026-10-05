"""Small loopback HTTP worker; the Rust frontend remains the public serving layer."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from models.cua_s1.multimodal.protocol import (
    MAX_BODY,
    InvalidRequest,
    MalformedJSON,
    decode_request,
    parse_request,
)

LOG = logging.getLogger(__name__)


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = False  # Join accepted handlers before retiring GPU buffers.

    def __init__(self, address, engine):
        self.engine = engine
        self.inference_lock = threading.Lock()
        self._engine_closed = False
        super().__init__(address, Handler)

    def server_close(self):
        super().server_close()
        if not self._engine_closed:
            close = getattr(self.engine, "close", None)
            if close is not None:
                close()
            self._engine_closed = True


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format, *args):
        # Do not log paths, input images, instructions or arbitrary request headers.
        pass

    def send_json(self, status, value):
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
            self.send_json(200, {"status": "ready", "modality": "multimodal"})
        else:
            self.send_json(404, {"detail": "unknown route"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self.send_json(404, {"detail": "unknown route"})
            return
        if self.headers.get("Transfer-Encoding"):
            self.send_json(
                411,
                {
                    "detail": "Content-Length is required; chunked requests are unsupported"
                },
            )
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
            parsed = parse_request(decode_request(raw))
            result = self.server.engine.predict(parsed)
            self.send_json(200, result)
        except MalformedJSON as exc:
            self.send_json(400, {"detail": str(exc)})
        except InvalidRequest as exc:
            self.send_json(422, {"detail": str(exc)})
        except (TimeoutError, socket.timeout):
            self.send_json(408, {"detail": "request body timed out"})
        except Exception as exc:
            LOG.error("inference failed: %s", type(exc).__name__)
            self.send_json(500, {"detail": "inference failed"})
        finally:
            self.server.inference_lock.release()


def parse_args(argv=None):
    from models.cua_s1.multimodal.graph_runtime import GraphConfig

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base", required=True, help="verified local base checkpoint directory"
    )
    p.add_argument(
        "--adapter", required=True, help="verified local multimodal adapter directory"
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--graph", action="store_true", help="enable segmented CUDA Graph replay"
    )
    p.add_argument(
        "--graph-mode",
        choices=("exact", "rule-bucket", "auto"),
        help="enable the selected Graph execution mode",
    )
    p.add_argument("--graph-bucket-width", type=int, default=None)
    p.add_argument("--graph-max-shapes", type=int, default=8)
    p.add_argument("--graph-max-memory-mib", type=int, default=1024)
    p.add_argument(
        "--graph-min-uses",
        type=int,
        default=2,
        help="distinct requests needed before capture",
    )
    p.add_argument("--graph-max-tokens", type=int, default=2048)
    p.add_argument("--graph-admission-window", type=int, default=8)
    p.add_argument("--graph-cooldown-requests", type=int, default=32)
    p.add_argument("--graph-capture-window", type=int, default=32)
    p.add_argument("--graph-max-captures", type=int, default=4)
    p.add_argument("--graph-capture-budget-ms", type=float, default=2000.0)
    args = p.parse_args(argv)
    mode = args.graph_mode or ("exact" if args.graph else None)
    if args.graph_bucket_width is not None and mode not in {"rule-bucket", "auto"}:
        p.error("--graph-bucket-width requires --graph-mode rule-bucket or auto")
    try:
        graph_config = (
            GraphConfig(
                mode=mode,
                bucket_width=64
                if args.graph_bucket_width is None
                else args.graph_bucket_width,
                max_shapes=args.graph_max_shapes,
                max_bytes=args.graph_max_memory_mib * 1024 * 1024,
                min_uses=args.graph_min_uses,
                max_tokens=args.graph_max_tokens,
                admission_window=args.graph_admission_window,
                cooldown_requests=args.graph_cooldown_requests,
                capture_window=args.graph_capture_window,
                max_captures=args.graph_max_captures,
                capture_budget_ms=args.graph_capture_budget_ms,
            )
            if mode is not None
            else None
        )
    except ValueError as exc:
        p.error(str(exc))
    args.graph_config = graph_config
    return args


def main():
    from models.cua_s1.multimodal.model import MultimodalEngine

    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    engine = MultimodalEngine(args.base, args.adapter, graph_config=args.graph_config)
    server = None
    try:
        engine.warmup()
        # Bind only after model loading and representative inference succeed.
        server = WorkerServer(("127.0.0.1", args.port), engine)
        LOG.info("multimodal worker ready on 127.0.0.1:%s", args.port)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        else:
            engine.close()


if __name__ == "__main__":
    main()
