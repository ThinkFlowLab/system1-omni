"""Shared, test-only helpers for the LFM2 recipe verification and benchmark scripts.

This module is not part of the model implementation. ``verify.py`` and
``bench.py`` use it to load the pinned RLCD reference engine, build mixed
engines that share one set of read-only weights, compute the independent
uncached oracle, and record a reproducibility manifest.
"""

import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "LiquidAI/LFM2.5-350M"
MODEL_ID = "LiquidAI/LFM2.5-350M"
MODEL_REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
REFERENCE_REPO = "notnotsamuel/LFM2.5-350M-RLCD"
REFERENCE_REVISION = "deb589d803d141cabd158ef55f6617b128529f36"
REFERENCE_ENGINE_SHA256 = "f6231cae0413fce75a5468533e44923cbc0ef796eb038e4c779fc149ec543736"
REFERENCE_URL = "https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE_DIR = DEFAULT_ROOT / "src" / "models" / "lfm2"


def repo_root():
    return DEFAULT_ROOT


def default_engine_dir():
    return DEFAULT_ENGINE_DIR


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    records = []
    path = Path(path)
    if not path.is_file():
        return records
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (quantile / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def load_module_from_path(name, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("module file not found: %s" % path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_target_engine(engine_dir):
    return load_module_from_path("lfm2_target_engine", Path(engine_dir) / "engine.py")


def load_target_worker(engine_dir):
    return load_module_from_path("lfm2_target_worker", Path(engine_dir) / "worker.py")


def load_reference(reference_dir):
    """Load the pinned RLCD checkout, accepting ``rlcd/engine.py`` or ``engine.py``."""
    root = Path(reference_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("reference dir not found: %s" % root)
    if (root / "rlcd" / "engine.py").is_file():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        engine = importlib.import_module("rlcd.engine")
        tasks = _optional_import("rlcd.tasks")
        stress = _optional_import("rlcd.stress_tasks")
        _check_reference_pins(engine, root)
        return SimpleNamespace(root=root, layout="rlcd-package", engine=engine,
                               tasks=tasks, stress_tasks=stress)
    if (root / "engine.py").is_file():
        engine = load_module_from_path("lfm2_reference_engine", root / "engine.py")
        tasks = stress = None
        if (root / "tasks.py").is_file():
            tasks = load_module_from_path("lfm2_reference_tasks", root / "tasks.py")
        if (root / "stress_tasks.py").is_file():
            stress = load_module_from_path("lfm2_reference_stress_tasks", root / "stress_tasks.py")
        _check_reference_pins(engine, root)
        return SimpleNamespace(root=root, layout="flat", engine=engine,
                               tasks=tasks, stress_tasks=stress)
    raise FileNotFoundError("no engine.py or rlcd/engine.py under %s" % root)


def _check_reference_pins(engine, root):
    if sha256_file(engine.__file__) != REFERENCE_ENGINE_SHA256:
        raise ValueError("reference engine differs from pinned revision " + REFERENCE_REVISION)
    module_model = getattr(engine, "MODEL_ID", None)
    module_revision = getattr(engine, "REVISION", None)
    if module_model is not None and module_model != MODEL_ID:
        raise ValueError("reference engine under %s pins model %s, expected %s"
                         % (root, module_model, MODEL_ID))
    if module_revision is not None and module_revision != MODEL_REVISION:
        raise ValueError("reference engine under %s pins revision %s, expected %s"
                         % (root, module_revision, MODEL_REVISION))


def _optional_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def load_shared_model(device, dtype, model_id=MODEL_ID, revision=MODEL_REVISION):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kwargs = {"attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype), **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=getattr(torch, dtype), **kwargs)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer


def build_reference_engine(reference, device, dtype, model, tokenizer):
    """Construct the pinned reference ``Engine`` against already-loaded weights.

    The pinned reference constructs its model and tokenizer internally; we
    temporarily swap the module's ``from_pretrained`` factories so the same
    read-only objects are reused instead of loading a second copy.
    """
    module = reference.engine

    class _SharedModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return model

    class _SharedTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return tokenizer

    saved_model = getattr(module, "AutoModelForCausalLM", None)
    saved_tokenizer = getattr(module, "AutoTokenizer", None)
    module.AutoModelForCausalLM = _SharedModel
    module.AutoTokenizer = _SharedTokenizer
    try:
        engine = module.Engine(device, dtype)
    finally:
        if saved_model is not None:
            module.AutoModelForCausalLM = saved_model
        if saved_tokenizer is not None:
            module.AutoTokenizer = saved_tokenizer
    engine.model = model
    engine.tokenizer = tokenizer
    return engine


def device_sync(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def candidate_count(schema):
    total = 0
    for spec in schema["properties"].values():
        total += 2 if spec.get("type") == "boolean" else len(spec.get("enum", []))
    return total


def field_specs(schema):
    for name, spec in schema["properties"].items():
        candidates = [True, False] if spec["type"] == "boolean" else list(spec["enum"])
        yield name, candidates


@torch.inference_mode()
def oracle_scores(engine, context, schema):
    """Independent uncached oracle: no shared cache and no candidate-batch scorer.

    For every candidate it concatenates ``prefix + suffix + value`` and runs a
    single full forward with ``use_cache=False``, then sums the FP32 log-softmax
    of the exact value token ids. This is deliberately separate from both the
    target ``Engine.score`` and the pinned reference ``Engine.constrained``.
    """
    prefix = engine.encode(engine.prompt(context, schema))
    scores = {}
    for name, candidates in field_specs(schema):
        suffix = engine.encode("  " + json.dumps(name, ensure_ascii=False) + ": ")
        options = []
        for candidate in candidates:
            value = engine.encode(json.dumps(candidate, ensure_ascii=False) + "\n")
            tokens = prefix + suffix + value
            start = len(prefix) + len(suffix)
            logits = engine.model(engine.tensor([tokens]), use_cache=False).logits[
                0, start - 1:start + len(value) - 1].float()
            score = logits.log_softmax(-1).gather(
                1, engine.tensor(value)[:, None]).sum().item()
            options.append({"value": candidate, "token_ids": list(value),
                            "log_likelihood": float(score)})
        scores[name] = options
    return {"scores": scores, "prompt_tokens": len(prefix), "prefix_token_ids": list(prefix)}


def selected_from(scores):
    return {name: max(options, key=lambda option: option["log_likelihood"])["value"]
            for name, options in scores.items()}


def margins_from(scores):
    margins = {}
    for name, options in scores.items():
        ordered = sorted((option["log_likelihood"] for option in options), reverse=True)
        margins[name] = ordered[0] - ordered[1] if len(ordered) > 1 else None
    return margins


def compare_scores(engine_scores, oracle_scores, schema, tolerance):
    maximum = 0.0
    per_field = {}
    selection_match = True
    for name in schema["properties"]:
        engine_options = {option["value"]: option for option in engine_scores[name]}
        oracle_options = {option["value"]: option for option in oracle_scores[name]}
        field_max = 0.0
        for value, oracle_option in oracle_options.items():
            if not math.isfinite(engine_options[value]["log_likelihood"]) or not math.isfinite(oracle_option["log_likelihood"]):
                raise ValueError("non-finite candidate score")
            difference = abs(engine_options[value]["log_likelihood"]
                             - oracle_option["log_likelihood"])
            field_max = max(field_max, difference)
        maximum = max(maximum, field_max)
        engine_selected = max(engine_options, key=lambda v: engine_options[v]["log_likelihood"])
        oracle_selected = max(oracle_options, key=lambda v: oracle_options[v]["log_likelihood"])
        selection_match = selection_match and engine_selected == oracle_selected
        per_field[name] = {"max_abs_error": field_max, "engine_selected": engine_selected,
                           "oracle_selected": oracle_selected}
    return {"max_abs_error": maximum, "within_tolerance": maximum < tolerance,
            "selection_match": selection_match, "fields": per_field}


def _nvidia_smi():
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False)
        if output.returncode == 0:
            return output.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _git_commit(path):
    try:
        output = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=30, check=False)
        if output.returncode == 0:
            return output.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def collect_environment(device, reference, source_paths, model_id=MODEL_ID,
                        revision=MODEL_REVISION):
    hashes = {}
    for path in source_paths:
        path = Path(path)
        if path.is_file():
            hashes[str(path)] = sha256_file(path)
    environment = {
        "recorded_at_utc": None,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "model_repo": model_id,
        "model_revision": revision,
        "reference_repo": REFERENCE_REPO,
        "reference_revision": REFERENCE_REVISION,
        "reference_layout": reference.layout,
        "reference_root": str(reference.root),
        "git_commit": _git_commit(repo_root()),
        "nvidia_smi": _nvidia_smi(),
        "source_sha256": hashes,
    }
    try:
        environment["transformers"] = importlib.import_module("transformers").__version__
    except Exception:
        environment["transformers"] = None
    if device.startswith("cuda") and torch.cuda.is_available():
        environment["gpu_name"] = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        environment["gpu_memory_bytes"] = int(properties.total_memory)
        environment["compute_capability"] = [int(properties.major), int(properties.minor)]
    return environment


def environment_id(environment):
    stable = {key: value for key, value in environment.items()
              if not key.endswith("_at_utc") and key != "recorded_at_utc"}
    return sha256_json(stable)


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
