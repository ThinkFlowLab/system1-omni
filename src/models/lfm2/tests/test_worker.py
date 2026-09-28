"""HTTP worker tests: request validation and response assembly with a fake engine.

The fake engine records the exact (context, schema) it receives, so prompt
independence from batching, ordering, renaming and the question id can be
checked without loading a model. A real loopback server exercises the
standard-library HTTP path in-process.
"""

import copy
import json
import math
import pathlib
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import worker as worker_mod  # noqa: E402

STATE = "I was charged twice. Please refund the duplicate today."
Q1 = {"type": "choice", "instructions": "Which team should handle this?",
      "criteria": {"billing": "Charges and refunds", "technical": "Software problems"}}
Q2 = {"type": "choice", "instructions": {"detail": "urgency"}, "criteria": {"low": None, "high": [1, 2]}}


class FakeEngine:
    def __init__(self, log_likelihoods=None, prompt_tokens=5):
        self.log_likelihoods = log_likelihoods or {}
        self.prompt_tokens = prompt_tokens
        self.calls = []

    def score(self, context, schema):
        self.calls.append((context, copy.deepcopy(schema)))
        name, spec = next(iter(schema["properties"].items()))
        values = spec["enum"]
        base = self.log_likelihoods.get(name, list(range(len(values))))
        entries = [{"value": value, "log_likelihood": float(base[i])}
                   for i, value in enumerate(values)]
        return {"scores": {name: entries}, "prompt_tokens": self.prompt_tokens,
                "telemetry": {"branches": len(values)}}


def payload(questions=None, **overrides):
    body = {"model": worker_mod.MODEL_ALIAS, "state": STATE,
            "questions": questions if questions is not None else {"refund": Q1}}
    body.update(overrides)
    return body


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(method, url, data=None):
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with OPENER.open(req, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def post(base, body, raw=False):
    data = body if raw else json.dumps(body, ensure_ascii=False).encode("utf-8")
    return request("POST", base + "/v1/systemone", data)


@pytest.fixture
def serve():
    servers = []

    def start(engine):
        server = worker_mod.make_server(worker_mod.Worker(engine), "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_health_and_response_contract(serve):
    engine = FakeEngine()
    base = serve(engine)
    assert request("GET", base + "/health") == (200, {"status": "ok"})

    status, body = post(base, payload())
    assert status == 200
    assert body["model"] == worker_mod.SERVED_MODEL
    assert body["usage"] == {"input_tokens": 5, "output_tokens": 0}
    answer = body["answers"]["refund"]
    assert answer["type"] == "choice"
    assert answer["choice"] == "technical"
    assert answer["probabilities"] == pytest.approx({"billing": 1 / (1 + math.e), "technical": math.e / (1 + math.e)})
    entropy = -(answer["probabilities"]["billing"] * math.log(answer["probabilities"]["billing"])
                + answer["probabilities"]["technical"] * math.log(answer["probabilities"]["technical"]))
    assert answer["confidence"] == pytest.approx(1 - entropy / math.log(2))
    assert len(engine.calls) == 1


def test_single_candidate_is_certain(serve):
    engine = FakeEngine()
    base = serve(engine)
    question = {"type": "choice", "instructions": "Only option?", "criteria": {"only": "one"}}
    status, body = post(base, payload(questions={"q": question}))
    assert status == 200
    answer = body["answers"]["q"]
    assert answer["choice"] == "only"
    assert answer["probabilities"] == {"only": 1.0}
    assert answer["confidence"] == 1.0


def test_ties_select_first_key(serve):
    base = serve(FakeEngine(log_likelihoods={"answer": [0.5, 0.5]}))
    status, body = post(base, payload())
    assert status == 200
    assert body["answers"]["refund"]["choice"] == "billing"


def test_prompt_uses_state_and_schema_only(serve):
    engine = FakeEngine()
    base = serve(engine)
    expected = worker_mod.build_schema(Q1["instructions"], Q1["criteria"])
    post(base, payload(questions={"q1": Q1}))
    post(base, payload(questions={"q2": Q2, "q1": Q1}))
    post(base, payload(questions={"renamed": Q1}))
    assert len(engine.calls) == 4
    for context, schema in engine.calls:
        assert context == STATE
        assert "q1" not in json.dumps(schema) and "renamed" not in json.dumps(schema)
    assert sum(schema == expected for _, schema in engine.calls) == 3
    assert engine.calls[0][0] == engine.calls[1][0] == engine.calls[2][0]


def test_instructions_and_criteria_shapes(serve):
    base = serve(FakeEngine(log_likelihoods={"answer": [0.0, 0.0, 0.0, 0.0]}))
    question = {"type": "choice", "instructions": ["a", "b"],
                "criteria": {"one": "text", "two": {"k": "v"}, "three": [1], "four": None}}
    status, body = post(base, payload(questions={"q": question}))
    assert status == 200
    assert body["answers"]["q"]["probabilities"] == pytest.approx(dict.fromkeys(question["criteria"], 0.25))
    for instructions in ("plain", {"k": "v"}, ["a", ["b"]]):
        question["instructions"] = instructions
        assert post(base, payload(questions={"q": question}))[0] == 200


@pytest.mark.parametrize("question", [
    {"type": "score", "instructions": "How urgent?", "criteria": ["a", "b"]},
    {"type": "noul", "instructions": "Refund?", "criteria": {"yes": "y"}},
    {"instructions": "Refund?", "criteria": {"yes": "y"}},
    {"type": "choice", "instructions": None, "criteria": {"yes": "y"}},
    {"type": "choice", "instructions": "Refund?", "criteria": {}},
    {"type": "choice", "instructions": "Refund?", "criteria": {"yes": 3}},
    {"type": "choice", "instructions": "Refund?",
     "criteria": {str(i): "v" for i in range(256)}},
])
def test_unsupported_requests_return_422(serve, question):
    base = serve(FakeEngine())
    assert post(base, payload(questions={"q": question}))[0] == 422


@pytest.mark.parametrize("body", [
    {"model": "other", "state": STATE, "questions": {"q": Q1}},
    {"model": worker_mod.MODEL_ALIAS, "state": "   ", "questions": {"q": Q1}},
    {"model": worker_mod.MODEL_ALIAS, "state": 7, "questions": {"q": Q1}},
    {"model": worker_mod.MODEL_ALIAS, "state": STATE, "questions": {}},
    {"model": worker_mod.MODEL_ALIAS, "state": STATE, "questions": []},
    {"model": worker_mod.MODEL_ALIAS, "state": STATE},
])
def test_bad_shapes_return_422(serve, body):
    base = serve(FakeEngine())
    assert post(base, body)[0] == 422


def test_malformed_json_returns_400(serve):
    base = serve(FakeEngine())
    assert post(base, b"{not json", raw=True)[0] == 400
    assert post(base, b"", raw=True)[0] == 400


def test_non_finite_and_duplicate_keys_return_400(serve):
    base = serve(FakeEngine())
    assert post(base, b'{"model":"lfm2.5-350m","state":NaN,"questions":{}}', raw=True)[0] == 400
    assert post(base, b'{"model":"lfm2.5-350m","state":Infinity,"questions":{}}', raw=True)[0] == 400
    body = json.dumps(payload()).replace(
        f'"model": "{worker_mod.MODEL_ALIAS}"',
        f'"model": "{worker_mod.MODEL_ALIAS}", "model": "other"')
    assert post(base, body.encode("utf-8"), raw=True)[0] == 400


def test_cli_requires_positive_candidate_batch_size():
    for argv in ([], ["--candidate-batch-size", "0"], ["--candidate-batch-size", "-2"],
                 ["--candidate-batch-size", "abc"]):
        with pytest.raises(SystemExit):
            worker_mod.parse_args(argv)
    args = worker_mod.parse_args(["--candidate-batch-size", "8"])
    assert args.candidate_batch_size == 8
    assert (args.device, args.dtype, args.host, args.port) == ("cuda", "float16", "127.0.0.1", 8000)


def test_independent_questions_keep_answers_and_ids(serve):
    engine = FakeEngine()
    base = serve(engine)
    first = post(base, payload(questions={"original": Q1}))[1]["answers"]["original"]
    for questions, qid in [
        ({"original": Q1, "unrelated": Q2}, "original"),
        ({"unrelated": Q2, "original": Q1}, "original"),
        ({"different-id": Q1}, "different-id"),
    ]:
        status, result = post(base, payload(questions=questions))
        assert status == 200
        assert set(result["answers"]) == set(questions)
        assert result["answers"][qid] == first


def test_invalid_later_question_does_not_run_earlier_one(serve):
    engine = FakeEngine()
    base = serve(engine)
    invalid = {"type": "noul", "instructions": "unsupported"}
    assert post(base, payload(questions={"valid": Q1, "invalid": invalid}))[0] == 422
    assert engine.calls == []
