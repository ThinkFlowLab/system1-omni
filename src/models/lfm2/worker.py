"""Standalone HTTP worker for LFM2.5-350M choice decisions.

Standard-library HTTP server (no web framework); requests are served serially so
GPU work never overlaps. The Rust frontend forwards `/health` and
`/v1/systemone` unchanged; see the frontend documentation for the transport
contract. Prompting, cache forking and candidate scoring live in engine.py,
adapted from notnotsamuel/LFM2.5-350M-RLCD (MIT, Copyright (c) 2026
notnotsamuel).
"""

import argparse
import json
import math
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ALIAS = "lfm2.5-350m"
MODEL_ID = "LiquidAI/LFM2.5-350M"
MODEL_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
SERVED_MODEL = f"{MODEL_ID}@{MODEL_REVISION}"
MAX_CRITERIA = 255

__all__ = ["Worker", "make_server", "main", "parse_args", "parse_json", "validate_request"]


class RequestError(Exception):
    """A client-visible 4xx error with a short message."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def parse_json(raw):
    """Decode a JSON object, rejecting duplicate keys and non-finite numbers."""
    def reject_constant(token):
        raise RequestError(400, f"invalid JSON constant: {token}")

    def reject_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise RequestError(400, "non-finite number is not allowed")
        return value

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RequestError(400, f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=no_duplicates,
                          parse_constant=reject_constant, parse_float=reject_float)
    except RequestError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RequestError(400, f"malformed JSON: {error}") from error


def _is_text_or_container(value):
    return isinstance(value, (str, dict, list))


def validate_request(payload, model_alias=MODEL_ALIAS):
    """Validate the whole request before any model work starts."""
    if not isinstance(payload, dict):
        raise RequestError(422, "request body must be a JSON object")
    if payload.get("model") != model_alias:
        raise RequestError(422, f"unsupported model; expected {model_alias!r}")
    state = payload.get("state")
    if not isinstance(state, str) or not state.strip():
        raise RequestError(422, "state must be a non-empty text string")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError(422, "questions must be a non-empty object")
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise RequestError(422, f"question {qid!r} must be an object")
        if question.get("type") != "choice":
            raise RequestError(422, f"question {qid!r} has unsupported type "
                                    f"{question.get('type')!r}; only 'choice' is supported")
        if not _is_text_or_container(question.get("instructions")):
            raise RequestError(422, f"question {qid!r} instructions must be a string, object or array")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_CRITERIA:
            raise RequestError(422, f"question {qid!r} criteria must map 1..{MAX_CRITERIA} keys")
        for value in criteria.values():
            if value is not None and not _is_text_or_container(value):
                raise RequestError(422, f"question {qid!r} criteria values must be "
                                        f"strings, objects, arrays or null")
    return payload


def build_schema(instructions, criteria):
    """One fixed `answer` field; enum keeps the original criteria key order."""
    description = json.dumps({"instructions": instructions, "criteria": criteria}, ensure_ascii=False)
    return {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": description,
                                  "enum": list(criteria.keys())}},
        "required": ["answer"],
        "additionalProperties": False,
    }


def softmax(logscores):
    if not all(math.isfinite(score) for score in logscores):
        raise ValueError("model returned non-finite candidate scores")
    largest = max(logscores)
    weights = [math.exp(score - largest) for score in logscores]
    total = sum(weights)
    return [weight / total for weight in weights]


def confidence(probabilities):
    count = len(probabilities)
    if count == 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(count)))


class Worker:
    """Turns one validated request into one response; holds the model engine."""

    def __init__(self, engine, model_alias=MODEL_ALIAS, served_model=SERVED_MODEL):
        self.engine = engine
        self.model_alias = model_alias
        self.served_model = served_model

    def health(self):
        return {"status": "ok"}

    def systemone(self, raw):
        payload = validate_request(parse_json(raw), self.model_alias)
        state = payload["state"]
        answers = {}
        input_tokens = 0
        for qid, question in payload["questions"].items():
            schema = build_schema(question["instructions"], question["criteria"])
            result = self.engine.score(state, schema)
            options = result["scores"]["answer"]
            logscores = [float(option["log_likelihood"]) for option in options]
            probabilities = softmax(logscores)
            index = max(range(len(logscores)), key=logscores.__getitem__)
            answers[qid] = {
                "type": "choice",
                "choice": options[index]["value"],
                "probabilities": {option["value"]: p for option, p in zip(options, probabilities)},
                "confidence": confidence(probabilities),
            }
            input_tokens += int(result.get("prompt_tokens", 0))
        return {
            "model": self.served_model,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "lfm2-worker/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";", 1)[0], 16)
                if size < 0:
                    raise ValueError("negative chunk size")
                if size == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                if self.rfile.read(2) != b"\r\n":
                    raise ValueError("invalid chunk terminator")
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0:
            raise ValueError("negative content length")
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self.send_json(200, self.server.worker.health())
        else:
            self.send_json(404, {"detail": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/v1/systemone":
            self.send_json(404, {"detail": "not found"})
            return
        try:
            raw = self.read_body()
        except (ValueError, OSError):
            self.send_json(400, {"detail": "invalid request body"})
            return
        try:
            response = self.server.worker.systemone(raw)
            status = 200
        except RequestError as error:
            response, status = {"detail": str(error)}, error.status
        except Exception:
            traceback.print_exc()
            response, status = {"detail": "internal error"}, 500
        try:
            self.send_json(status, response)
        except (BrokenPipeError, ConnectionResetError):
            pass


class WorkerServer(HTTPServer):
    daemon_threads = False

    def __init__(self, address, worker):
        self.worker = worker
        super().__init__(address, Handler)


def make_server(worker, host="127.0.0.1", port=8000):
    return WorkerServer((host, port), worker)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate-batch-size", type=_positive_int, required=True,
                        help="candidates scored per forward pass (required, no default)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def _positive_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_engine(args):
    from engine import Engine
    return Engine(candidate_batch_size=args.candidate_batch_size, device=args.device, dtype=args.dtype)


def main(argv=None):
    args = parse_args(argv)
    engine = build_engine(args)
    server = make_server(Worker(engine), args.host, args.port)
    sys.stderr.write("lfm2 worker listening on http://%s:%d\n" % server.server_address[:2])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
