import json
import math

import pytest

from frontend.valen import decide
from models.valen.engine import ExecutorOutput
from models.valen.preprocess import CompiledInput
from models.valen.protocol import DEFAULT_SPECIAL_TOKENS, RequestError

from .test_protocol import body


class FakeProcessor:
    special_tokens = DEFAULT_SPECIAL_TOKENS

    def __init__(self):
        self.seen_media = None

    def compile(self, prepared):
        if prepared.image_path is not None:
            assert prepared.image_path.is_file()
            self.seen_media = prepared.image_path
        return CompiledInput(object(), prepared.response_context)


class RejectingProcessor(FakeProcessor):
    def compile(self, prepared):
        raise ValueError("Reserved tokenizer control token in state")


class FakeExecutor:
    def __init__(self, output):
        self.output = output

    def execute(self, compiled):
        assert compiled.response_context.questions
        return self.output

    def close(self):
        pass


def test_decide_keeps_http_adapter_outside_executor():
    processor = FakeProcessor()
    executor = FakeExecutor(ExecutorOutput(((0.0, 1.0),), 17, 17))
    result = decide(json.dumps(body()).encode("utf-8"), processor, executor)

    assert result["model"] == "valen-preview-0923"
    assert result["answers"]["move"]["choice"] == "down"
    assert result["usage"] == {"input_tokens": 17, "output_tokens": 0}
    assert result["internal_usage"] == {"compute_tokens": 17}
    assert processor.seen_media is not None
    assert not processor.seen_media.exists()


def test_decide_serves_text_only_state_without_media():
    processor = FakeProcessor()
    executor = FakeExecutor(ExecutorOutput(((0.0, 1.0),), 12, 12))
    value = body(state="The card was charged twice for one order.")
    result = decide(json.dumps(value).encode("utf-8"), processor, executor)

    assert result["answers"]["move"]["choice"] == "down"
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 0}
    assert result["internal_usage"] == {"compute_tokens": 12}
    assert processor.seen_media is None


def test_decide_rejects_executor_cardinality_mismatch():
    processor = FakeProcessor()
    executor = FakeExecutor(ExecutorOutput((), 1, 1))

    try:
        decide(json.dumps(body()).encode("utf-8"), processor, executor)
    except RuntimeError as error:
        assert "answer count" in str(error)
    else:
        raise AssertionError("executor cardinality mismatch was accepted")


def test_decide_maps_pinned_compiler_value_errors_to_422():
    executor = FakeExecutor(ExecutorOutput(((0.0, 1.0),), 1, 1))
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(body()).encode("utf-8"), RejectingProcessor(), executor)

    assert excinfo.value.status == 422
    assert "pinned compiler" in str(excinfo.value)
    assert "Reserved tokenizer control token" in str(excinfo.value)


def test_decide_keeps_request_errors_unwrapped():
    class WireErrorProcessor(FakeProcessor):
        def compile(self, prepared):
            raise RequestError("wire problem", 422)

    executor = FakeExecutor(ExecutorOutput(((0.0, 1.0),), 1, 1))
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(body()).encode("utf-8"), WireErrorProcessor(), executor)

    assert excinfo.value.status == 422
    assert str(excinfo.value) == "wire problem"


def test_decide_answers_mixed_type_questions_from_one_state():
    value = body(
        state="The card was charged twice for one order.",
        questions={
            "ask": {"type": "noul", "instructions": "Was the card charged twice?"},
            "grade": {
                "type": "score",
                "instructions": "Grade the severity.",
                "criteria": ["No error", "A minor error", "A major error"],
            },
            "act": {
                "type": "choice",
                "instructions": "Decide the refund action.",
                "criteria": {"refund": None, "wait": "Wait for review"},
            },
        },
    )
    processor = FakeProcessor()
    executor = FakeExecutor(
        ExecutorOutput(
            (
                (math.log(3.0), 0.0),  # noul: true, false
                (math.log(7.0), math.log(2.0), 0.0),  # score levels 0, 1, 2
                (0.0, 1.0),  # choice: refund, wait
            ),
            61,
            183,
        )
    )

    result = decide(json.dumps(value).encode("utf-8"), processor, executor)

    noul = result["answers"]["ask"]
    assert noul["type"] == "noul"
    assert noul["noul"] == pytest.approx(0.75)
    assert noul["probabilities"]["true"] == pytest.approx(0.75)
    assert noul["probabilities"]["false"] == pytest.approx(0.25)

    score = result["answers"]["grade"]
    assert score["type"] == "score"
    assert score["probabilities"]["0"] == pytest.approx(0.7)
    assert score["score"] == pytest.approx(0.4)
    assert score["legend"] == {"0": "No error", "1": "A minor error", "2": "A major error"}
    assert score["confidence"] == pytest.approx(0.4)

    assert result["answers"]["act"]["choice"] == "wait"
    assert result["usage"] == {"input_tokens": 61, "output_tokens": 0}
    assert result["internal_usage"] == {"compute_tokens": 183}

