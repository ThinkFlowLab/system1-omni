"""Strict CUDA-only official Laya candidates; no accuracy equivalence is implied."""

import functools
import importlib.metadata
import math

REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
VARIANTS = (
    "stock_fp32",
    "stock_bf16",
    "fast_no_graph",
    "fast_graph",
    "fast_full_graph",
)


class FastPathError(Exception):
    """Intentionally not RuntimeError: Laya must not retry failed CUDA work on CPU."""


def install_guards(router, agent, variant):
    """Preserve official computation while rejecting fallback and changed dispatch."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    expected_fast = agent._fast
    original_forward = agent.model.forward
    original_predict = router.predict
    first_parameter = next(agent.model.parameters())
    expected_device = str(first_parameter.device)
    expected_dtype = str(agent.dtype)
    expected_amp = bool(agent.amp_enabled)
    is_fast = variant.startswith("fast_")
    if is_fast:
        cls = type(expected_fast)
        official_forward = (
            getattr(original_forward, "_official_fast_forward", None)
            if variant == "fast_full_graph"
            else original_forward
        )
        if (
            cls.__module__ != "laya.fast"
            or cls.__name__ != "FastLaya"
            or getattr(official_forward, "__self__", None) is not expected_fast
        ):
            raise FastPathError(
                "Official FastLaya is absent or forward is not bound to it"
            )
        if expected_fast.use_graphs != (variant == "fast_graph"):
            raise FastPathError("Official graph setting does not match variant")
        if str(expected_fast.layers[0]["wqkv"].dtype) != "torch.bfloat16":
            raise FastPathError("Official fast QKV weights are not BF16")
    elif expected_fast is not None:
        raise FastPathError("Stock variant unexpectedly has FastLaya installed")

    def check():
        parameter = next(agent.model.parameters())
        if (
            agent.device.type != "cuda"
            or parameter.device.type != "cuda"
            or str(parameter.device) != expected_device
            or str(agent.dtype) != expected_dtype
            or bool(agent.amp_enabled) != expected_amp
        ):
            raise FastPathError("CUDA device/AMP contract changed; fallback forbidden")
        if agent._fast is not expected_fast:
            raise FastPathError("FastLaya identity changed; fallback forbidden")
        if is_fast and (
            str(expected_fast.dev) != expected_device
            or expected_fast.use_graphs != (variant == "fast_graph")
        ):
            raise FastPathError("FastLaya device/graph contract changed")

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        try:
            check()
            result = original_forward(*args, **kwargs)
            check()
            return result
        except FastPathError:
            raise
        except Exception as error:
            raise FastPathError(
                f"{variant} forward failed: {type(error).__name__}: {error}"
            ) from error

    @functools.wraps(original_predict)
    def predict(*args, **kwargs):
        check()
        if agent.model.forward is not forward:
            raise FastPathError("Model forward dispatch changed")
        result = original_predict(*args, **kwargs)
        check()
        if agent.model.forward is not forward:
            raise FastPathError("Model forward dispatch changed")
        return result

    def no_fallback():
        raise FastPathError(
            "Agent attempted deaccelerate/CPU fallback; request aborted"
        )

    check()
    agent.model.forward = forward
    agent.deaccelerate = no_fallback
    router.predict = predict
    agent._benchmark_variant = variant
    agent._benchmark_check = check
    agent._benchmark_original_forward = original_forward
    return router, agent


def make_router(variant):
    """Create one frozen checkpoint, using official accelerate(strict=True) for fast."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    import torch
    from huggingface_hub import snapshot_download
    from laya import Agent, Router

    if importlib.metadata.version("laya") != "0.3.20":
        raise FastPathError("This candidate requires frozen laya==0.3.20")
    if not torch.cuda.is_available():
        raise FastPathError("CUDA unavailable; no CPU fallback permitted")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = snapshot_download(
        "convaiinnovations/laya",
        revision=REVISION,
        local_files_only=True,
        allow_patterns=[
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        ],
    )
    agent = Agent(path, device="cuda", fast=False, compile=False)
    if (
        agent.device.type != "cuda"
        or next(agent.model.parameters()).device.type != "cuda"
    ):
        raise FastPathError("Checkpoint loading fell back from CUDA")
    agent.amp_enabled = variant != "stock_fp32"
    agent.dtype = torch.float32 if variant == "stock_fp32" else torch.bfloat16
    if variant.startswith("fast_"):
        if (
            agent.accelerate(use_graphs=variant == "fast_graph", strict=True)
            is not True
        ):
            raise FastPathError("Official accelerate did not report success")
        if variant == "fast_full_graph":
            from full_graph_candidate import install_full_graph

            install_full_graph(agent)
    router = Router(device="cuda", max_loaded=1)
    router.attach("english", agent)
    return install_guards(router, agent, variant)


def metadata(agent):
    """Dispatch and precision evidence; called outside measured requests."""
    import torch

    agent._benchmark_check()
    fast = agent._fast
    versions = {}
    for package in ("laya", "torch", "transformers", "huggingface_hub", "tilelang"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "variant": agent._benchmark_variant,
        "revision": REVISION,
        "device": str(agent.device),
        "hardware": torch.cuda.get_device_name(agent.device),
        "parameter_devices": sorted({str(p.device) for p in agent.model.parameters()}),
        "parameter_dtypes": sorted({str(p.dtype) for p in agent.model.parameters()}),
        "amp_enabled": agent.amp_enabled,
        "amp_dtype": str(agent.dtype),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "versions": versions,
        "cpu_threads": torch.get_num_threads(),
        "fallback_policy": "hard failure; official CPU retry blocked",
        "fast": None,
    }
    if fast is not None:
        original = agent._benchmark_original_forward
        if agent._benchmark_variant == "fast_full_graph":
            original = original._official_fast_forward
        result["fast"] = {
            "class": f"{type(fast).__module__}.{type(fast).__name__}",
            "original_forward_bound_to_fast": getattr(original, "__self__", None)
            is fast,
            "device": str(fast.dev),
            "use_graphs": fast.use_graphs,
            "graph_scope": "encoder + decision transformer; scorer/action outside graph",
            "embedding_dtype": str(fast.emb_w.dtype),
            "qkv_weight_dtype": str(fast.layers[0]["wqkv"].dtype),
            "graph_shapes": [list(key) for key in fast.graphs],
            "max_len": fast.max_len,
        }
        if agent._benchmark_variant == "fast_full_graph":
            from full_graph_candidate import metadata as full_graph_metadata

            result["full_graph"] = full_graph_metadata(agent)
    return result


def validate_response(response, request):
    """Schema/finite checks only; numerical drift is recorded, not silently accepted."""

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise FastPathError("Non-finite response")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                finite(child)

    def number(value, low, high):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise FastPathError(f"Invalid numeric response field: {value!r}")

    finite(response)
    try:
        if (
            response["model"] != "laya-rl-agent"
            or response["routing"]["model"] != "english"
            or set(response["answers"]) != set(request["questions"])
        ):
            raise FastPathError("Wrong model/routing/question ids")
        usage = response["usage"]
        if (
            type(usage["output_tokens"]) is not int
            or usage["output_tokens"] != 0
            or type(usage["input_tokens"]) is not int
            or usage["input_tokens"] < 1
        ):
            raise FastPathError("Invalid complete-response usage")
        for qid, question in request["questions"].items():
            answer = response["answers"][qid]
            kind = question["type"]
            if answer["type"] != kind:
                raise FastPathError("Question type changed")
            for field in ("confidence", "answer_confidence"):
                number(answer[field], 0, 1)
            number(answer["action"]["act_probability"], 0, 1)
            if kind == "noul":
                number(answer["noul"], 0, 1)
            else:
                keys = (
                    set(question["criteria"])
                    if kind == "choice"
                    else {str(i) for i in range(len(question["criteria"]))}
                )
                probabilities = answer["probabilities"]
                if set(probabilities) != keys:
                    raise FastPathError("Missing probabilities")
                for value in probabilities.values():
                    number(value, 0, 1)
                if abs(sum(probabilities.values()) - 1) > len(keys) * 0.0001:
                    raise FastPathError(
                        "Probabilities do not sum to one within rounding"
                    )
                if kind == "choice" and answer["choice"] not in keys:
                    raise FastPathError("Invalid choice")
                if kind == "score":
                    number(answer["score"], 0, len(keys) - 1)
                    if answer["legend"] != {
                        str(i): v for i, v in enumerate(question["criteria"])
                    }:
                        raise FastPathError("Score legend changed")
    except (KeyError, TypeError, AttributeError) as error:
        raise FastPathError(f"Incomplete official response schema: {error}") from error
