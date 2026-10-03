"""HTTP worker for Laya on Apple Silicon (PyTorch MPS) and CPU: `GET /health` and `POST /v1/systemone`.

PYTHONPATH=src python -m frontend.laya_mps --device mps --model english [--compile] [--weights fp16]

It is laya-serve (`laya[serve]==0.3.20`) with its request handling unchanged and three changes:

- It binds only after every loaded model has run a warmup over short, long and multi-question requests,
  so a reachable worker is a warm one. laya-serve answers /health before any forward pass. A checkpoint
  laya loads later (a request that names another model, or routes to it) gets the same options and
  warmup inside that first request.
- /health describes the loaded models as they are now: device, weight and autocast dtypes, checkpoint
  and the revision the weights were loaded from, and `device_mismatch` when a model is not on the
  requested device. laya-serve reports the configured device, and laya moves a model to the CPU on a
  GPU out-of-memory error and keeps serving. `--require-device` exits at startup instead of serving
  from another device.
- `--compile` and `--weights fp16` make the GPU path faster (models/laya/optimize.py). Both apply on
  the GPU only; on the CPU, including after a fallback, the worker runs laya's fp32 model uncompiled.

Of laya-serve's environment variables, the ones its app reads still apply, notably LAYA_API_KEY for bearer
authentication. The ones its launcher reads do not, because the flags above replace it: LAYA_DEVICE,
LAYA_MODELS, LAYA_PRELOAD, LAYA_HOST, LAYA_PORT, LAYA_LOG_LEVEL, LAYA_THREADS and LAYA_AUTO_TASK. The
worker warns at startup about any of those that are set.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from models.laya import engine, optimize

log = logging.getLogger("laya-worker")

# Read by laya-serve's launcher (laya.serve.build_router and main), which this worker does not run.
UNREAD_LAYA_SERVE_VARIABLES = (
    "LAYA_DEVICE",
    "LAYA_MODELS",
    "LAYA_PRELOAD",
    "LAYA_HOST",
    "LAYA_PORT",
    "LAYA_LOG_LEVEL",
    "LAYA_THREADS",
    "LAYA_AUTO_TASK",
)


class Lifecycle:
    """laya Router hooks that hand checkpoint loads and evictions to one worker app."""

    def __init__(self, on_load, on_evict):
        self.on_load = on_load
        self.on_evict = on_evict


def build_app(
    router: Any,
    model: str,
    requested: str | None,
    *,
    require_device: bool = False,
    compile: bool = False,
    fp16: bool = False,
    graph_counter=optimize.compiled_graphs,
    revisions: dict[str, str] | None = None,
):
    """Prepare every loaded model (options, warmup, device check), then return laya's app with /health
    replaced. A checkpoint laya loads later, for a request that names or routes to it, is prepared the same
    way before it answers. `model` is the one summarised at the top of /health, and the one loaded if
    nothing is preloaded. Raises if preparing fails, so the caller never binds a worker that cannot answer."""
    from laya.serve import create_app

    resident: dict[str, tuple[Any, dict[str, Any]]] = {}  # prepared checkpoints: name -> (agent, warmup result)
    preparing: set[str] = set()
    graphs_at_ready = None

    def apply_options(name: str, agent: Any) -> None:
        if (fp16 or compile) and not optimize.apply(agent, fp16=fp16, compile=compile):
            log.warning("%s is on the CPU: --compile and --weights fp16 apply on the GPU only", name)

    def make_ready(name: str, agent: Any) -> str | None:
        """Warm the checkpoint up and start describing it. Returns where it is if not on the requested device."""
        nonlocal graphs_at_ready
        warmed = engine.warmup(router, name)
        warmed["revision"] = engine.loaded_revision(revisions, warmed["routing"])  # fixed here: see engine
        resident[name] = (agent, warmed)
        autocast_rows = getattr(agent, "mps_amp_min_rows", None)
        if str(agent.device).startswith("mps") and autocast_rows and autocast_rows > engine.WARMUP_MAX_ROWS:
            log.warning(
                "%s: laya autocasts from %d questions but the warmup stops at %d; the first request that "
                "large is not warm",
                name,
                autocast_rows,
                engine.WARMUP_MAX_ROWS,
            )
        if compile:
            graphs_at_ready = graph_counter()
        described = engine.describe(agent, requested, warmed["routing"], warmed["revision"])
        return f"{name} is on {described['device']}" if described["device_mismatch"] else None

    def check_device(misplaced: list[str]) -> None:
        if misplaced:
            message = f"asked for {requested}, {', '.join(misplaced)}"
            if require_device:
                raise RuntimeError(message)
            log.warning(message)

    evicted: list[str] = []  # what laya dropped to make room for the checkpoint it is loading

    def on_evict(ctx: Any) -> None:
        resident.pop(ctx.model, None)
        evicted.append(ctx.model)

    def on_load(ctx: Any) -> None:
        """A checkpoint loaded while serving is prepared like the ones loaded at startup, or not kept at all;
        in that case the checkpoints laya evicted for it are loaded again."""
        nonlocal graphs_at_ready
        made_room = [name for name in evicted if name != ctx.model]
        evicted.clear()
        preparing.add(ctx.model)
        try:
            apply_options(ctx.model, ctx.agent)
            check_device(list(filter(None, [make_ready(ctx.model, ctx.agent)])))
        except Exception:
            log.exception("%s could not be prepared and is unloaded", ctx.model)
            router.unload(ctx.model)
            evicted.clear()
            for name in made_room:
                try:
                    router.load(name)
                except Exception:  # noqa: BLE001 -- the request fails for the first reason either way
                    log.exception("%s was evicted for %s and could not be loaded again", name, ctx.model)
            if compile:
                graphs_at_ready = graph_counter()  # graphs the unloaded checkpoint compiled are not recompiles
            raise
        finally:
            preparing.discard(ctx.model)

    names = list(router.loaded) or [model]  # never load a model the worker was not asked to serve
    startup = {name: router.load(name) for name in names}
    for name, agent in startup.items():
        apply_options(name, agent)
    check_device(list(filter(None, [make_ready(name, agent) for name, agent in startup.items()])))
    for hook in [h for h in getattr(router, "hooks", ()) if isinstance(h, Lifecycle)]:
        router.remove_hook(hook)  # an app built earlier on this router
    router.add_hook(Lifecycle(on_load, on_evict))

    def current() -> dict[str, Any]:
        """The agents as they are now, not as they were at startup (see the module docstring)."""
        models = {
            name: {
                **engine.describe(agent, requested, warmed["routing"], warmed["revision"]),
                "warmup_ms": warmed["warmup_ms"],
            }
            for name, (agent, warmed) in list(resident.items())
        }
        return {
            **models.get(model, next(iter(models.values()), {})),
            "device_mismatch": any(m["device_mismatch"] for m in models.values()),
            "warmup_ms": round(sum(m["warmup_ms"] for m in models.values()), 1),
            "models": models,
        }

    app = create_app(router)
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/health"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        compiled: dict[str, Any] = {"enabled": compile}
        if compile:
            now = graph_counter()
            compiled.update(
                active=any(optimize.compile_active(agent) for agent, _ in list(resident.values())),
                graphs_at_ready=graphs_at_ready,
                graphs_now=now,
                recompiled_after_ready=now > graphs_at_ready and not preparing,
            )
        return {
            "status": "ok",
            "ready": True,
            "loaded": router.loaded,
            "preparing": sorted(preparing),
            **current(),
            "compile": compiled,
        }

    return app


def make_router(device: str | None, model: str) -> Any:
    from laya.router import Router

    router = Router(device=device)
    router.preload([model])
    return router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="torch device for laya: mps, cpu (default: laya's choice)")
    parser.add_argument("--model", default="english", help="laya checkpoint to serve: english, multilingual, ...")
    parser.add_argument("--compile", action="store_true", help="torch.compile the GPU path during warmup")
    parser.add_argument("--weights", default="fp32", choices=["fp32", "fp16"], help="weight precision on the GPU")
    parser.add_argument("--require-device", action="store_true", help="exit if a model is not on --device")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="for the worker's own log and uvicorn's",
    )
    args = parser.parse_args()

    import uvicorn

    logging.basicConfig(level=args.log_level.upper(), format="%(name)s: %(message)s")
    unread = [name for name in UNREAD_LAYA_SERVE_VARIABLES if os.environ.get(name)]
    if unread:
        log.warning("%s: read by laya-serve's launcher, not by this worker; use the flags", ", ".join(unread))
    revisions = engine.record_snapshot_revisions()
    try:
        app = build_app(
            make_router(args.device, args.model),
            args.model,
            args.device,
            require_device=args.require_device,
            compile=args.compile,
            fp16=args.weights == "fp16",
            revisions=revisions,
        )
    except Exception as exc:  # noqa: BLE001 -- any failure before binding means not ready, ever
        log.exception("startup failed")
        sys.exit(f"laya-worker: not starting: {exc}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
