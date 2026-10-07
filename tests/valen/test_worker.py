import json

from frontend.valen import decide
from models.valen.engine import ExecutorOutput
from models.valen.preprocess import CompiledInput

from .test_protocol import body


class FakeProcessor:
    def __init__(self):
        self.seen_media = None

    def compile(self, prepared):
        if prepared.image_path is not None:
            assert prepared.image_path.is_file()
            self.seen_media = prepared.image_path
        return CompiledInput(object(), prepared.response_context)


class FakeExecutor:
    def __init__(self, output):
        self.output = output

    def execute(self, compiled):
        assert compiled.response_context.questions
        return self.output


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

