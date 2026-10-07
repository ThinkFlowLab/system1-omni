import base64

import pytest
from PIL import Image

from models.valen.protocol import (
    MODEL_NAME,
    MalformedJSON,
    RequestError,
    parse_body,
    parse_request,
)


def image_data_url(format_name="PNG"):
    from io import BytesIO

    stream = BytesIO()
    Image.new("RGB", (2, 2), "white").save(stream, format_name)
    mime = "png" if format_name == "PNG" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(stream.getvalue()).decode()}"


def body(**overrides):
    value = {
        "model": MODEL_NAME,
        "state": {"image": image_data_url()},
        "questions": {
            "move": {
                "type": "choice",
                "instructions": "Choose a move.",
                "criteria": {"up": "Move up", "down": "Move down"},
            }
        },
    }
    value.update(overrides)
    return value


def test_parse_valid_inline_image_and_choice():
    request = parse_request(body())

    assert request.model == MODEL_NAME
    assert request.state.format == "PNG"
    assert request.state.width == request.state.height == 2
    assert request.questions[0].keys == ("up", "down")
    assert request.questions[0].descriptions == ("Move up", "Move down")


def test_parse_accepts_text_state_in_every_wire_shape():
    parsed = parse_request(body(state="Refund the duplicated charge."))
    assert parsed.state.text == "Refund the duplicated charge."

    parsed = parse_request(body(state={"goal": "route", "lang": "zh"}))
    assert parsed.state.text == '{"goal": "route", "lang": "zh"}'

    parsed = parse_request(body(state=["part one", "part two"]))
    assert parsed.state.text == '["part one", "part two"]'


def test_parse_body_rejects_duplicate_and_nonfinite_json():
    with pytest.raises(MalformedJSON, match="duplicate"):
        parse_body(b'{"model":"x","model":"y"}')
    with pytest.raises(MalformedJSON, match="non-finite"):
        parse_body(b'{"value":NaN}')


def test_parse_rejects_wrong_model_and_invalid_state():
    with pytest.raises(RequestError, match="model must"):
        parse_request(body(model="Valen"))
    with pytest.raises(RequestError, match="must not be empty"):
        parse_request(body(state=""))
    with pytest.raises(RequestError, match="exactly one image"):
        parse_request(body(state={"image": "data:image/png;base64,AAA", "extra": 1}))
    with pytest.raises(RequestError, match="image data URL or text"):
        parse_request(body(state=17))


def test_parse_allows_structured_text_but_not_score_or_extra_fields():
    value = body()
    value["questions"]["move"]["instructions"] = {"goal": "choose"}
    assert parse_request(value).questions[0].instructions == '{"goal": "choose"}'

    value = body()
    value["questions"]["move"]["type"] = "score"
    with pytest.raises(RequestError, match="only choice"):
        parse_request(value)

    value = body()
    value["questions"]["move"]["unexpected"] = True
    with pytest.raises(RequestError, match="unsupported fields"):
        parse_request(value)


def test_parse_accepts_jpeg_and_rejects_mismatched_mime():
    value = body(state={"image": image_data_url("JPEG")})
    assert parse_request(value).state.format == "JPEG"

    value = body(state={"image": image_data_url("JPEG").replace("image/jpeg", "image/png")})
    with pytest.raises(RequestError, match="does not match"):
        parse_request(value)


def test_parse_rejects_media_control_tokens():
    value = body()
    value["questions"]["move"]["instructions"] = "<|image_pad|>"
    with pytest.raises(RequestError, match="media control token"):
        parse_request(value)

    with pytest.raises(RequestError, match="media control token"):
        parse_request(body(state="context <|video_pad|> tail"))
