"""HTTP tests for the Valen worker handler: status mapping and body limits.

Starts the real WorkerServer on an ephemeral port with the fake processor and
executor from test_worker: no weights, no torch, no GPU. The 408 slow-client
timeout is not covered because the handler's 15 s connection timeout is
deliberately not configurable for tests.

    PYTHONPATH=src python -m pytest tests/valen -q
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest

from frontend.valen import WorkerServer
from models.valen.engine import ExecutorOutput
from models.valen.protocol import MAX_BODY

from .test_protocol import body
from .test_worker import FakeExecutor, FakeProcessor


class FailingExecutor:
    def execute(self, compiled):
        raise RuntimeError("boom")

    def close(self):
        pass


@pytest.fixture()
def server():
    return start(FakeExecutor(ExecutorOutput(((0.0, 1.0),), 17, 17)))


def start(executor):
    server = WorkerServer(("127.0.0.1", 0), FakeProcessor(), executor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def stop(server):
    server.shutdown()
    server.server_close()


def request(server, method, path, headers=(), payload=b""):
    """One raw HTTP request with full header control."""

    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest(method, path, skip_accept_encoding=True)
    for name, value in headers:
        conn.putheader(name, value)
    conn.endheaders(payload)
    response = conn.getresponse()
    raw = response.read()
    status, ctype = response.status, response.getheader("Content-Type")
    conn.close()
    return status, ctype, raw


def post(server, payload, headers=()):
    """POST /v1/systemone with JSON defaults; explicit headers replace them."""

    defaults = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
    extra = []
    for name, value in headers:
        defaults.pop(name, None)
        extra.append((name, value))
    return request(
        server,
        "POST",
        "/v1/systemone",
        headers=(*defaults.items(), *extra),
        payload=payload,
    )


def test_health_and_unknown_get_route(server):
    status, ctype, raw = request(server, "GET", "/health")
    assert status == 200 and ctype == "application/json"
    health = json.loads(raw)
    assert health["status"] == "ready"
    assert health["model"] == "valen-preview-0923"

    status, _, raw = request(server, "GET", "/nope")
    assert status == 404 and json.loads(raw) == {"detail": "unknown route"}


def test_valid_decision_returns_the_answer(server):
    status, ctype, raw = post(server, json.dumps(body()).encode("utf-8"))
    assert status == 200 and ctype == "application/json"
    result = json.loads(raw)
    assert result["answers"]["move"]["choice"] == "down"
    assert result["usage"] == {"input_tokens": 17, "output_tokens": 0}


def test_unknown_post_route_is_404(server):
    status, _, _ = request(
        server,
        "POST",
        "/other",
        headers=(("Content-Type", "application/json"), ("Content-Length", "2")),
        payload=b"{}",
    )
    assert status == 404


def test_length_headers_are_enforced(server):
    payload = b"{}"
    base = {"Content-Type": "application/json"}

    status, _, raw = post(server, payload, headers=(("Transfer-Encoding", "chunked"),))
    assert status == 411 and "Content-Length" in json.loads(raw)["detail"]

    status, _, _ = request(
        server,
        "POST",
        "/v1/systemone",
        headers=[*base.items(), ("Content-Length", "2"), ("Content-Length", "2")],
        payload=payload,
    )
    assert status == 411

    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/v1/systemone", skip_accept_encoding=True)
    conn.putheader("Content-Type", "application/json")
    conn.endheaders(payload)
    response = conn.getresponse()
    response.read()
    assert response.status == 411
    conn.close()

    status, _, _ = post(server, payload, headers=(("Content-Length", "abc"),))
    assert status == 400


def test_body_limit_and_content_type_are_enforced(server):
    payload = json.dumps(body()).encode("utf-8")

    status, _, raw = post(server, payload, headers=(("Content-Length", str(MAX_BODY + 1)),))
    assert status == 413 and "body limit" in json.loads(raw)["detail"]

    status, _, raw = post(server, payload, headers=(("Content-Type", "text/plain"),))
    assert status == 415 and "application/json" in json.loads(raw)["detail"]


def test_client_errors_map_to_their_status_and_detail(server):
    status, _, raw = post(server, b"{not json")
    assert status == 400 and "detail" in json.loads(raw)

    value = body(model="Valen")
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422 and "model must" in json.loads(raw)["detail"]

    value = body()
    value["questions"]["move"]["instructions"] = "  "
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422 and "nonempty" in json.loads(raw)["detail"]


def test_busy_worker_returns_503(server):
    server.inference_lock.acquire()
    try:
        status, _, raw = post(server, json.dumps(body()).encode("utf-8"))
    finally:
        server.inference_lock.release()
    assert status == 503 and json.loads(raw) == {"detail": "worker busy"}


def test_executor_failure_returns_500_without_leaking_details():
    failing = start(FailingExecutor())
    try:
        status, _, raw = post(failing, json.dumps(body()).encode("utf-8"))
    finally:
        stop(failing)
    assert status == 500 and json.loads(raw) == {"detail": "inference failed"}
