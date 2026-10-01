"""HTTP tests for the worker with a fake model: no weights, no torch."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from frontend.cua_s1_text import build_app  # noqa: E402


class Ids:
    def __init__(self, n):
        self.shape = (1, n)


class FakeModel:
    def __init__(self, tokens=100, error=None, nan=False):
        self.tokens, self.error, self.nan, self.calls = tokens, error, nan, 0

    def encode(self, state, question):
        return {"input_ids": Ids(self.tokens)}

    def score(self, inputs, n_options):
        self.calls += 1
        if self.error:
            raise self.error
        p = [0.1] * (n_options - 1) + [1 - 0.1 * (n_options - 1)]
        return [float("nan")] + p[1:] if self.nan else p


BODY = {
    "model": "cua-s1-4b-0.2",
    "state": "Screen",
    "questions": {
        "_sa": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"_x": "A", "b": "B"},
        }
    },
}


def post(model=None, **kwargs):
    return TestClient(build_app(model or FakeModel())).post("/v1/systemone", **kwargs)


def test_health_and_answer():
    app = build_app(FakeModel())
    assert TestClient(app).get("/health").json()["status"] == "ready"
    response = post(json=BODY)
    assert response.status_code == 200, response.text
    reply = response.json()
    assert reply["answers"]["_sa"]["choice"] == "b"
    assert list(reply["answers"]["_sa"]["probabilities"]) == ["_x", "b"]
    assert reply["usage"] == {"input_tokens": 100, "output_tokens": 0}


def test_errors():
    bad = json.loads(json.dumps(BODY))
    bad["questions"]["_sa"]["type"] = "noul"
    assert post(json=bad).status_code == 422
    assert post(content=b"{").status_code == 400
    raw = b" " * (4 << 20) + json.dumps(BODY).encode()
    assert post(content=iter([raw[:10], raw[10:]])).status_code == 413
    model = FakeModel(tokens=20000)
    assert post(model, json=BODY).status_code == 413 and model.calls == 0


@pytest.mark.parametrize(
    "model", [FakeModel(error=RuntimeError("CUDA out of memory")), FakeModel(nan=True)]
)
def test_model_failure_is_500(model):
    response = post(model, json=BODY)
    assert response.status_code == 500
    assert response.json() == {"detail": "inference failed"}


def test_warmup():
    model = FakeModel()
    build_app(model).state.warmup()
    assert model.calls == 1
