"""D3 executor for the pinned code-readout checkpoint.

The executor owns the pinned runtime, the manifest verification that gates its
import, device state, and warmup. It consumes raw state/questions through the
upstream ``prepare``/``run``/``respond`` surface and stays free of HTTP
concerns; wire admission lives in protocol.py and preprocess.py.

The checkpoint ships its own runtime as flat Python files next to the
weights. Those files are the code this process executes, so — stronger than
the upstream default, whose ``verify_package`` walks only the 39-entry
``files_sha256`` table and skips hashing files over 64 MiB in fast mode — the
executor hashes every file of the manifest's ``runtime.files_sha256`` table
in full before importing anything from the package directory, and refuses
unlisted importable entries (an extra ``*.py`` file or ``__init__.py``
package) in the package root: that root goes first on ``sys.path``, so such
an entry would shadow a dependency the pinned runtime imports. Hash
agreement proves only internal consistency — a later snapshot ships a
self-consistent table of its own — so the gate first requires the manifest
to carry exactly the repo id and runtime digests checked in for
``REFERENCE_REVISION``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

REFERENCE_REPO = "vllm-sr/d3"
REFERENCE_REVISION = "cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75"
MANIFEST_NAME = "MODEL_MANIFEST.json"
MANIFEST_SCHEMA = "d3-package-manifest/1"
VERIFY_MODES = ("fast", "full", "none")
# The runtime.files_sha256 table of REFERENCE_REVISION, transcribed from the
# pinned package's manifest; repinning means updating this table together
# with REFERENCE_REVISION.
PINNED_RUNTIME_SHA256 = {
    "d3_runtime.py": "6c049aa7d3da8da3738602a2812572879d0a47b03c28c3e33f36d7300257de89",
    "d3_fast.py": "68a1fb9a2c3bde9ac2763bf504decae592ca12722698fbde1b39a31262955535",
    "d3_kernels.py": "8f7534ca1e1846ebf7c4c73058002f3be4dbf0d5d426f0815150ae0cbe7677bd",
    "modeling_d3.py": "f8f8db23ff58f1a990ffa24977d744a50e1b4446d02a64c22070967705b6dbef",
    "pipeline_d3.py": "4f9297c7c74e596e9b67df91a1e6823cfada48bf781336f141974ff6a2f1038e",
    "d3_server.py": "f9b8f2481461a074f469b4f58da3b0d7004824096528366d5b1f23187dd6e692",
    "d3_engine.py": "086238942762df45047a32cd9a8e2bedd9b63167eb40be260c776de567979f53",
    "requirements.txt": "57a1938448bc240bdf4fee5d1b9cbc8450109fa1f6cee11e94dc04f274be8a32",
    "d3_format.py": "e5036154d2e54793f59b320c8726632957bc343c382bac26d825b263a6e9ce62",
}


def _sha256_safe(path: Path) -> str:
    """Might encounter a large file, read in chunks"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> dict[str, Any]:
    """Read and schema-check the package manifest; refuse packages without one."""

    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(
            f"the package is missing {MANIFEST_NAME}; this worker only serves verified packages"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{MANIFEST_NAME} is not a {MANIFEST_SCHEMA} manifest")
    return manifest


def verify_pinned_package(manifest: dict[str, Any]) -> None:
    """Refuse any package that is not the pinned reference snapshot.

    A package whose files hash to its own manifest is only internally
    consistent: a later snapshot rewrites its ``runtime.files_sha256`` and
    would pass the full-hash gate on a changed baseline. The manifest must
    therefore carry exactly the repo id and runtime digests checked in for
    ``REFERENCE_REVISION`` before any file is hashed or imported.
    """

    repo_id = manifest.get("repo_id")
    if repo_id != REFERENCE_REPO:
        raise ValueError(
            f"{MANIFEST_NAME} declares repo_id {repo_id!r}; this worker only "
            f"serves {REFERENCE_REPO} at {REFERENCE_REVISION}"
        )
    table = (manifest.get("runtime") or {}).get("files_sha256")
    if table != PINNED_RUNTIME_SHA256:
        raise ValueError(
            f"{MANIFEST_NAME} runtime.files_sha256 is not the table pinned for "
            f"{REFERENCE_REPO}@{REFERENCE_REVISION}; re-pin deliberately or "
            "download the pinned revision"
        )


def verify_runtime_files(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Hash every runtime file of the manifest in full and return their names.

    Must run before any package code is imported: the runtime table names the
    files this process is about to execute, so unlike the upstream shard
    verification there is no size shortcut. It also refuses importable
    entries the table does not name — the package root is about to go first
    on ``sys.path``, where an extra ``*.py`` file or ``__init__.py`` package
    would shadow one of the pinned runtime's dependencies.
    """

    table = (manifest.get("runtime") or {}).get("files_sha256")
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{MANIFEST_NAME} has no runtime.files_sha256 table")
    problems: list[str] = []
    for name, digest in sorted(table.items()):
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"unsafe runtime file name {name!r}")
        target = Path(root) / name
        if not target.is_file():
            problems.append(f"missing runtime file {name}")
            continue
        if _sha256_safe(target) != digest:
            problems.append(f"{name}: runtime SHA-256 differs from {MANIFEST_NAME}")
    for entry in sorted(Path(root).iterdir()):
        if entry.name in table:
            continue
        if entry.is_file() and entry.suffix == ".py":
            problems.append(f"unexpected importable file {entry.name} in the package root")
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            problems.append(f"unexpected importable package {entry.name}/ in the package root")
    if problems:
        raise ValueError("runtime verification failed: " + "; ".join(problems[:8]))
    return sorted(table)


class D3Executor:
    """Load and drive one immutable pinned d3 runtime instance."""

    model_name = "d3"

    def __init__(
        self,
        package: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 8,
        verify: str = "fast",
        model_name: str | None = None,
        max_length: int | None = None,
        readout_dtype: str | None = None,
        permutation_average: bool = False,
    ) -> None:
        if verify not in VERIFY_MODES:
            raise ValueError(f"verify must be one of {VERIFY_MODES}")
        root = Path(package)
        if not root.is_dir():
            raise ValueError(f"the d3 package directory does not exist: {root}")
        manifest = load_manifest(root)
        verify_pinned_package(manifest)
        verify_runtime_files(root, manifest)

        # Import the just-verified runtime the way upstream d3_server.py does:
        # the package directory goes first on sys.path so the flat runtime
        # modules (d3_runtime, d3_format, ...) resolve to the checkpoint's own
        # copies. One worker process serves one package.
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        import d3_runtime

        self.model = d3_runtime.D3.from_pretrained(
            root,
            device=device,
            batch_size=batch_size,
            verify=verify,
            model_name=model_name,
            max_length=max_length,
            readout_dtype=readout_dtype,
            permutation_average=permutation_average,
        )
        self.model_name = self.model.model_name
        self.max_length = self.model.max_length
        self._image_unavailable = self.model.image_unavailable
        self._video_unavailable = self.model.video_unavailable
        self._lock = threading.RLock()
        self._closed = False

    @property
    def image_unavailable(self) -> str | None:
        """Why the loaded checkpoint cannot read images; ``None`` when it can.

        Mirrors the runtime's own flag: it is set when the vision tower
        weights, the patch embedding or the image processor are missing, and
        the upstream server refuses non-empty ``images`` on it.
        """

        return self._image_unavailable

    @property
    def video_unavailable(self) -> str | None:
        """Why the loaded checkpoint cannot read videos; ``None`` when it can."""

        return self._video_unavailable

    @property
    def modalities(self) -> list[str]:
        """The loaded model's input modalities, in upstream /health order."""

        names = ["text", "image"] if self._image_unavailable is None else ["text"]
        return names + ["video"] if self._video_unavailable is None else names

    def load_images(self, images: list[Any]) -> list[Any]:
        """Decode one request's images (RGB PIL) through strict loading.

        The pinned runtime is the per-item validation authority — data URL
        shape, PNG/JPEG/WEBP format, the 8,000,000-byte and 16,000,000-pixel
        budgets — and its ``ValueError`` rejections name the item
        (``images[i]: ...``). The empty short-circuit and the capability gate
        with the upstream server's message live in the worker's decode stage,
        mirroring ``d3_server.py``'s ``decode_images``.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.load_images(images, strict=True)

    def load_videos(self, videos: list[Any]) -> list[Any]:
        """Decode and sample one request's videos through strict loading.

        Per-item validation (MP4/WebM/QuickTime/Matroska data URLs, the
        32,000,000-byte, 300-second and per-frame budgets, 2 fps sampling of
        4-32 frames) is the pinned runtime's; rejections name the item
        (``videos[i]: ...``).
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.load_videos(videos, strict=True)

    def prepare(
        self,
        state: Any,
        questions: dict[str, Any],
        images: Any = None,
        videos: Any = None,
    ) -> Any:
        """Validate and tokenize one request; per-question failures land in
        ``prepared.errors`` instead of raising (envelope problems raise).

        ``images``/``videos`` are already-decoded media from the worker's
        decode stage; the runtime picks the video, image or text path by
        their truthiness, exactly as the upstream server passes them.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.prepare(state, questions, images, videos)

    def run(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        """Probabilities per runnable question and the input token count."""

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.run(prepared)

    def respond(self, prepared: Any, probabilities: dict[str, list[float]], tokens: int) -> dict:
        """Build the System One response, embedding per-question model failures."""

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            return self.model.respond(prepared, probabilities, tokens)

    def warmup(self) -> None:
        """Compile and autotune the kernels before readiness."""

        with self._lock:
            if self._closed:
                raise RuntimeError("D3 executor is closed")
            self.model.warmup()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.model = None
