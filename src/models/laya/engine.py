"""Laya as a served model: what to run before readiness and what to report about a loaded model.

laya's `Router` and `Agent` do the loading and the forward passes. This module adds the warmup every
loaded model runs before the worker binds, the description of a loaded model that `/health` returns, and
the record of which checkpoint revision laya actually loaded. The HTTP worker is `frontend/laya_mps.py`;
the GPU optimizations are `optimize.py` next to this file.
"""

import time
from pathlib import Path
from typing import Any

# (words of state, questions). Short, mid-length and near-window states, then several questions; the last
# shape has enough rows to reach laya's fp16 autocast on MPS (`agent.mps_amp_min_rows`).
_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges and refunds", "technical": "Software problems", "other": "Anything else"},
}
_SCORE = {"type": "score", "instructions": "How urgent is it?", "criteria": ["Low", "Medium", "High"]}
_NOUL = {"type": "noul", "instructions": "Does the customer ask for a refund?"}
WARMUP_SHAPES = [
    (10, {"q": _CHOICE}),
    (150, {"q": _CHOICE}),
    (400, {"q": _CHOICE}),
    (10, {"a": _CHOICE, "b": _SCORE, "c": _NOUL}),
    (10, {f"q{i}": q for i, q in enumerate([_CHOICE, _SCORE, _NOUL, _CHOICE, _SCORE, _NOUL])}),
]
WARMUP_REPEATS = 2
WARMUP_MAX_ROWS = max(len(questions) for _, questions in WARMUP_SHAPES)


def warmup(router: Any, model: str, shapes=WARMUP_SHAPES, repeats: int = WARMUP_REPEATS) -> dict[str, Any]:
    """Any failure propagates: a worker that cannot answer must not bind."""
    started = time.perf_counter()
    routing = None
    for words, questions in shapes:
        state = " ".join(["refund"] * words)
        for _ in range(repeats):
            result = router.predict(state, questions, model=model)
            routing = result.get("routing") or routing
    return {"warmup_ms": round((time.perf_counter() - started) * 1000, 1), "routing": routing}


def _checkpoint_name(repo_id: str, allow_patterns: Any) -> str:
    """laya's name for what one download fetched: "<repo>", or "<repo>/<subfolder>" for a bundled checkpoint.
    laya restricts each download to one checkpoint's files, which all sit under its subfolder if it has one."""
    patterns = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns or [""])
    folders = {pattern.split("/")[0] if "/" in pattern else "" for pattern in patterns}
    subfolder = folders.pop() if len(folders) == 1 else ""
    return f"{repo_id}/{subfolder}" if subfolder else repo_id


def record_snapshot_revisions() -> dict[str, str]:
    """Record the commit each Hugging Face checkpoint was last downloaded at, keyed by laya's name for it
    (`routing["repo"]`).

    laya calls huggingface_hub.snapshot_download while loading and keeps only the repo id; the
    returned path (.../snapshots/<commit>/...) is the only place the loaded revision appears. Call this
    before the router loads anything. A checkpoint loaded from a local path records nothing. An entry
    changes when the same checkpoint is downloaded again, so read it with `loaded_revision` right after a
    checkpoint has loaded and keep that value.
    """
    import huggingface_hub

    revisions: dict[str, str] = {}
    original = huggingface_hub.snapshot_download

    def recording(repo_id, *args, **kwargs):
        path = original(repo_id, *args, **kwargs)
        parts = Path(path).parts
        if "snapshots" in parts[:-1]:
            name = _checkpoint_name(repo_id, kwargs.get("allow_patterns"))
            revisions[name] = parts[parts.index("snapshots") + 1]
        return path

    huggingface_hub.snapshot_download = recording
    return revisions


def loaded_revision(revisions: dict[str, str] | None, routing: dict[str, Any] | None) -> str | None:
    """The commit of the checkpoint that has just loaded, if it was downloaded."""
    return (revisions or {}).get((routing or {}).get("repo"))


def describe(
    agent: Any, requested: str | None, routing: dict[str, Any] | None, revision: str | None = None
) -> dict[str, Any]:
    """What /health reports about one loaded agent, read from the agent as it is now."""
    device = str(getattr(agent, "device", "unknown"))
    model = getattr(agent, "model", None)
    try:
        weights = str(next(model.parameters()).dtype) if model is not None else None
    except (AttributeError, StopIteration, TypeError):
        weights = None
    repo = (routing or {}).get("repo")
    requested_type = requested.split(":")[0] if requested else None
    return {
        "device": device,
        "requested_device": requested or "auto",
        "device_mismatch": bool(requested_type) and device.split(":")[0] != requested_type,
        "weights_dtype": weights,
        "autocast_dtype": str(getattr(agent, "dtype", None)),
        "mps_amp_min_rows": getattr(agent, "mps_amp_min_rows", None),
        "checkpoint": repo,
        "revision": revision,
    }
