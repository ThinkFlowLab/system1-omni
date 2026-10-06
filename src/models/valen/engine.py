"""Valen executor for the pinned Preview checkpoint.

The executor owns the reference model, learned decision head, forward pass,
device state, and warmup. It consumes compiled model inputs and returns logits
and token accounting only; API probabilities and response reconstruction remain
in postprocess.py.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .preprocess import CompiledInput
from .protocol import MODEL_NAME

REFERENCE_REVISION = "750bfcfbb48a5275534a9c912257ebe83ca57a97"
CHECKPOINT_REVISION = "81b9c63"
CHECKPOINT_SHA256 = (
    "836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012"
)
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
EXPECTED_STAGE = "vision_top"
EXPECTED_PROJECTION_DIM = 256
EXPECTED_MAX_LENGTH = 8192


@dataclass(frozen=True)
class ExecutorOutput:
    logits: tuple[tuple[float, ...], ...]
    logical_tokens: int
    compute_tokens: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ValenExecutor:
    """Load and execute one immutable Valen Preview model instance."""

    model_name = MODEL_NAME

    def __init__(
        self,
        checkpoint: str | Path,
        base: str | Path,
        device: str = "cuda",
        dtype: str = "bf16",
    ) -> None:
        checkpoint_path = Path(checkpoint).resolve()
        base_path = Path(base).resolve()
        config_path = checkpoint_path / "config.json"
        weights_path = checkpoint_path / "checkpoint.pt"
        if not config_path.is_file() or not weights_path.is_file():
            raise ValueError("checkpoint must contain config.json and checkpoint.pt")
        if _sha256(weights_path) != CHECKPOINT_SHA256:
            raise ValueError("Valen-Preview-0923 checkpoint checksum mismatch")
        if not base_path.is_dir():
            raise ValueError(f"Qwen base directory does not exist: {base_path}")
        if dtype not in {"bf16", "fp32"}:
            raise ValueError("dtype must be bf16 or fp32")
        self._validate_base_manifest(base_path)

        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._validate_config(config)
        config["model_path"] = str(base_path)
        config["device"] = device
        config["dtype"] = dtype

        try:
            from valen.modeling.model import build_model
            from valen.training.checkpoint import load_checkpoint
        except ImportError as exc:
            raise RuntimeError(
                "install the pinned Valen source and reference dependencies before starting"
            ) from exc

        self.model = build_model(config)
        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _validate_base_manifest(base_path: Path) -> None:
        manifest_path = base_path / "valen_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Qwen base is missing valen_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("repo") != "Qwen/Qwen3.5-2B"
            or manifest.get("revision") != BASE_REVISION
        ):
            raise ValueError("Qwen base manifest does not match the pinned revision")

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        if config.get("stage") != EXPECTED_STAGE:
            raise ValueError(f"checkpoint stage must be {EXPECTED_STAGE}")
        if config.get("projection_dim") != EXPECTED_PROJECTION_DIM:
            raise ValueError("checkpoint projection_dim does not match Preview")
        if config.get("max_length") != EXPECTED_MAX_LENGTH:
            raise ValueError("checkpoint max_length does not match Preview")

    def execute(self, compiled: CompiledInput) -> ExecutorOutput:
        """Run all question forwards serially and return CPU FP32 logits."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Valen executor is closed")
            import torch

            state = compiled.state
            rows: list[tuple[float, ...]] = []
            with torch.no_grad():
                for question in state.questions:
                    logits = self.model(question)
                    if logits.ndim != 1:
                        raise RuntimeError("Valen executor returned non-vector logits")
                    rows.append(tuple(float(value) for value in logits.float().cpu().tolist()))
            return ExecutorOutput(
                tuple(rows),
                int(state.logical_tokens),
                int(state.compute_tokens),
            )

    def warmup(self, compiled: CompiledInput) -> None:
        """Run a real representative compile and forward before readiness."""

        self.execute(compiled)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.model = None

