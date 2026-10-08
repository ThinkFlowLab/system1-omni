import hashlib

from models.valen.preprocess import prepare_request
from models.valen.protocol import MODEL_NAME, parse_request

from .test_protocol import body


def test_prepare_materializes_request_scoped_image_and_valen_record(tmp_path):
    request = parse_request(body())
    prepared = prepare_request(request, tmp_path)

    expected_hash = hashlib.sha256(request.state.data).hexdigest()
    assert prepared.image_path == tmp_path / "input.png"
    assert prepared.image_path.read_bytes() == request.state.data
    assert prepared.image_sha256 == expected_hash
    assert prepared.record["assets"] == [{"path": "input.png", "sha256": expected_hash}]
    assert prepared.record["request"]["state"] == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "input.png"}}
                ],
            }
        ]
    }
    assert prepared.record["request"]["questions"]["move"]["criteria"] == {
        "up": "Move up",
        "down": "Move down",
    }
    assert prepared.response_context.model == MODEL_NAME
    assert [q.name for q in prepared.response_context.questions] == ["move"]


def test_prepare_compiles_text_state_without_media(tmp_path):
    request = parse_request(body(state="Dialog: the update finished."))
    prepared = prepare_request(request, tmp_path)

    assert prepared.image_path is None
    assert prepared.image_sha256 is None
    assert prepared.record["assets"] == []
    assert prepared.record["request"]["state"] == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Dialog: the update finished."}
                ],
            }
        ]
    }
    assert list(prepared.record["request"]["questions"]) == ["move"]
    assert list(tmp_path.iterdir()) == []
