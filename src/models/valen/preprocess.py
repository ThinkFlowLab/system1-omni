"""Valen-specific prepared-input and compiler processing.

The public contract carries an inline image for compatibility with Cua-S1.
Valen's pinned compiler requires a local media path and an optional audited
hash, so this layer materializes one request-scoped file and produces the
upstream record shape. The worker owns the lifetime of media_root.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import Request, Question


@dataclass(frozen=True)
class ResponseContext:
    model: str
    questions: tuple[Question, ...]


@dataclass(frozen=True)
class PreparedInput:
    record: dict[str, Any]
    media_root: Path
    image_path: Path
    image_sha256: str
    response_context: ResponseContext


@dataclass(frozen=True)
class CompiledInput:
    """Model-specific compiled state plus the response identity context."""

    state: Any
    response_context: ResponseContext


class ValenProcessor:
    """Own the pinned tokenizer/processor and Valen compiler boundary."""

    def __init__(
        self,
        base: str | Path,
        max_length: int,
        media_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from transformers import AutoProcessor
            from valen.data.compiler import Compiler
        except ImportError as exc:
            raise RuntimeError(
                "install the pinned Valen source and reference dependencies before starting"
            ) from exc
        self._compiler_type = Compiler
        self._max_length = max_length
        self._media_kwargs = media_kwargs or {}
        self.processor = AutoProcessor.from_pretrained(
            str(base), local_files_only=True
        )

    def compile(self, prepared: PreparedInput) -> CompiledInput:
        """Compile one prepared request without owning model/device state."""

        compiler = self._compiler_type(
            self.processor,
            media_root=prepared.media_root,
            max_length=self._max_length,
            media_kwargs=self._media_kwargs,
        )
        return CompiledInput(
            compiler.compile(prepared.record),
            prepared.response_context,
        )


def prepare_request(request: Request, media_root: Path) -> PreparedInput:
    """Materialize one validated request into the pinned Valen record format."""

    media_root = Path(media_root)
    media_root.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if request.image.format == "PNG" else ".jpg"
    image_path = media_root / f"input{suffix}"
    with image_path.open("xb") as stream:
        stream.write(request.image.data)
    digest = hashlib.sha256(request.image.data).hexdigest()

    criteria = {
        question.name: {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": dict(zip(question.keys, question.descriptions)),
        }
        for question in request.questions
    }
    record = {
        "request": {
            "state": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_path.name},
                            }
                        ],
                    }
                ]
            },
            "questions": criteria,
        },
        "assets": [{"path": image_path.name, "sha256": digest}],
    }
    return PreparedInput(
        record=record,
        media_root=media_root,
        image_path=image_path,
        image_sha256=digest,
        response_context=ResponseContext(request.model, request.questions),
    )

