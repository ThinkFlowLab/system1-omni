"""HTTP tests for the d3 worker handler: status mapping and body limits.

Starts the real WorkerServer on an ephemeral port with the FakeD3 from
test_worker: no weights, no torch, no GPU. The 408 slow-client timeout is not
covered because the handler's 15 s connection timeout is deliberately not
configurable for tests. Strict video decode (OpenCV against the pinned
runtime) cannot run here; the media cases exercise the decode stage's
admission, capability and rejection surface only.

    PYTHONPATH=src python -m pytest tests/d3 -q
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from http.client import HTTPConnection
from io import BytesIO

import pytest
from PIL import Image

from frontend.d3 import WorkerServer
from models.d3.protocol import MAX_BODY

from .test_protocol import body
from .test_worker import FakeD3, three_kind_body


def png_data_url():
    """One small real PNG as an inline data URL (valen-test precedent)."""

    stream = BytesIO()
    Image.new("RGB", (2, 2), "white").save(stream, "PNG")
    return f"data:image/png;base64,{base64.b64encode(stream.getvalue()).decode()}"


class FailingExecutor:
    model_name = "d3"
    max_length = None
    modalities = ["text"]
    image_unavailable = "the checkpoint has no vision tower (visual.* weights)"
    video_unavailable = image_unavailable

    def load_images(self, images):
        raise RuntimeError("boom")

    def load_videos(self, videos):
        raise RuntimeError("boom")

    def prepare(self, state, questions):
        raise RuntimeError("boom")

    def close(self):
        pass


def start(executor):
    server = WorkerServer(("127.0.0.1", 0), executor)
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


@pytest.fixture()
def server():
    instance = start(FakeD3())
    yield instance
    stop(instance)


def test_health_reports_model_limits_and_modalities(server):
    status, ctype, raw = request(server, "GET", "/health")
    assert status == 200 and ctype == "application/json"
    health = json.loads(raw)
    assert health["status"] == "ready"
    assert health["model"] == "d3"
    assert health["max_input_tokens"] is None
    assert health["modalities"] == ["text"]
    assert "authentication" not in health

    status, _, raw = request(server, "GET", "/nope")
    assert status == 404 and json.loads(raw) == {"detail": "unknown route"}


def test_health_reports_three_modalities_for_a_vision_checkpoint():
    vision = start(FakeD3(media=True))
    try:
        status, _, raw = request(vision, "GET", "/health")
    finally:
        stop(vision)
    assert status == 200
    health = json.loads(raw)
    assert health["status"] == "ready"
    assert health["modalities"] == ["text", "image", "video"]


def test_valid_three_kind_decision_returns_200(server):
    status, ctype, raw = post(server, json.dumps(three_kind_body()).encode("utf-8"))
    assert status == 200 and ctype == "application/json"
    result = json.loads(raw)
    assert result["model"] == "d3"
    assert result["answers"]["urgent"] == {"type": "noul", "noul": 0.5}
    assert result["usage"]["output_tokens"] == 0


def test_media_request_returns_200_on_a_vision_checkpoint():
    vision = start(FakeD3(media=True))
    try:
        value = three_kind_body()
        value["images"] = [png_data_url()]
        status, ctype, raw = post(vision, json.dumps(value).encode("utf-8"))
    finally:
        stop(vision)
    assert status == 200 and ctype == "application/json"
    result = json.loads(raw)
    assert result["model"] == "d3"
    assert result["answers"]["urgent"] == {"type": "noul", "noul": 0.5}


def test_undecodable_media_rejects_the_whole_request():
    vision = start(FakeD3(media=True))
    try:
        value = three_kind_body()
        value["images"] = ["data:image/png;base64,!!!!"]
        status, _, raw = post(vision, json.dumps(value).encode("utf-8"))
    finally:
        stop(vision)
    assert status == 422
    assert json.loads(raw) == {"detail": "images[0]: invalid base64 image data"}


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


def test_truncated_body_is_rejected(server):
    # The client declares 100 bytes, sends 7 and half-closes its write side:
    # the worker must answer 400 (incomplete body) on the readable socket.
    import socket

    sock = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=10)
    sock.sendall(
        b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n"
        b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{short"
    )
    sock.shutdown(socket.SHUT_WR)
    response = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk
    sock.close()
    status_line = response.split(b"\r\n", 1)[0]
    assert b" 400 " in status_line
    assert b"incomplete body" in response


def test_body_limit_and_content_type_are_enforced(server):
    payload = json.dumps(body()).encode("utf-8")

    status, _, raw = post(server, payload, headers=(("Content-Length", str(MAX_BODY + 1)),))
    assert status == 413 and "body limit" in json.loads(raw)["detail"]
    assert "64 MiB" in json.loads(raw)["detail"]

    status, _, raw = post(server, payload, headers=(("Content-Type", "text/plain"),))
    assert status == 415 and "application/json" in json.loads(raw)["detail"]


def test_transport_errors_map_to_their_status_and_detail(server):
    status, _, raw = post(server, b"{not json")
    assert status == 400 and "detail" in json.loads(raw)

    value = body(extra=True)
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422 and "Unknown request fields" in json.loads(raw)["detail"]

    # A non-list media field is refused with the upstream server's message.
    value = body()
    value["images"] = "data:image/png;base64,AAAA"
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422
    assert json.loads(raw)["detail"] == "images must be a list of base64 data URLs"

    # Non-empty media on the text-only fixture is refused with the upstream
    # capability messages (empty lists would take the text path).
    value = body()
    value["images"] = [png_data_url()]
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422
    assert json.loads(raw)["detail"] == "This model reads text only; images are not supported."

    value = body()
    value["videos"] = ["data:video/mp4;base64,AAAA"]
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422
    assert json.loads(raw)["detail"] == "This model does not read videos."


def test_invalid_questions_return_the_upstream_422_body(server):
    value = three_kind_body()
    value["questions"]["broken"] = {"type": "triangle"}
    status, _, raw = post(server, json.dumps(value).encode("utf-8"))
    assert status == 422
    payload = json.loads(raw)
    assert set(payload) == {"invalid_questions"}
    assert "unsupported question type" in payload["invalid_questions"]["broken"]


def test_partial_success_returns_200_and_logs_a_warning(server, caplog):
    executor = FakeD3(probabilities={"refund": [1.0, 3.0], "urgent": [0.0, 0.0]})
    partial = start(executor)
    try:
        with caplog.at_level(logging.WARNING, logger="models.d3.postprocess"):
            status, ctype, raw = post(
                partial, json.dumps(three_kind_body()).encode("utf-8")
            )
    finally:
        stop(partial)

    assert status == 200 and ctype == "application/json"
    result = json.loads(raw)
    broken = result["answers"]["urgent"]
    assert broken["error"] == "invalid_model_output"
    assert result["answers"]["refund"]["choice"] == "wait"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "urgent" in warnings[0].getMessage()


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
