#!/usr/bin/env python3
"""Run local, reserved-GPU HTTP accuracy/latency regression checks."""

import argparse
import asyncio
import contextlib
import io
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

if __package__:
    from . import bench
else:
    import bench

ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = ROOT
CONTROL_KEYS = (
    "gpu",
    "gpu_ids",
    "driver",
    "cuda",
    "precision",
    "model_revision",
    "cache_policy",
    "host",
    "cpu_affinity",
    "gpu_uuids",
    "harness_sha256",
)


def load_suite(path):
    suite = json.loads(path.read_text())
    if not suite.get("models"):
        raise ValueError("suite needs models")
    for key in ("concurrency", "warmup", "request_timeout", "startup_timeout"):
        if key in suite and (type(suite[key]) is not int or suite[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in (
        "max_accuracy_drop",
        "max_score_mae_increase",
        "max_latency_increase",
        "max_probability_drift",
        "max_score_drift",
        "max_flips",
    ):
        value = suite["gates"][key]
        if not bench.number(value) or value < 0:
            raise ValueError(f"invalid gate {key}")
    if type(suite["gates"]["max_flips"]) is not int:
        raise ValueError("max_flips must be an integer")
    ids = [model["id"] for model in suite["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model IDs")
    for model in suite["models"]:
        if (
            not model["id"]
            or Path(model["id"]).name != model["id"]
            or model["id"] in (".", "..")
        ):
            raise ValueError("model ID must be a directory name")
        if model.get("unavailable"):
            continue
        if model.get("reference_model") and model["reference_model"] not in ids:
            raise ValueError(f"{model['id']}: unknown reference_model")
        for key in ("health", "endpoint"):
            if not isinstance(model.get(key), str) or not model[key]:
                raise ValueError(f"{model['id']}: requires {key}")
        cases = bench.load_cases(ROOT / model["manifest"])
        for case in cases:
            bench.wire_request(case, model["model"], model.get("protocol", "systemone"))
        for key in CONTROL_KEYS[:7]:
            if not model["metadata"].get(key):
                raise ValueError(f"{model['id']}: metadata requires {key}")
        if model.get("protocol", "systemone") not in ("systemone", "jev-vl"):
            raise ValueError("unknown protocol")
        if type(model.get("score_decimals", 4)) is not int or model.get(
            "score_decimals", 4
        ) not in (2, 3, 4):
            raise ValueError("score_decimals must match the model contract (2, 3 or 4)")
        if not model.get("commands"):
            raise ValueError(f"{model['id']}: commands must launch worker and frontend")
        if "CUDA_VISIBLE_DEVICES" in model.get("env", {}):
            raise ValueError("let the GPU scheduler configure CUDA_VISIBLE_DEVICES")
        for command in model["commands"]:
            if (
                not isinstance(command, list)
                or not command
                or not all(isinstance(s, str) for s in command)
            ):
                raise ValueError("commands must be argument lists, without a shell")
            executable = command[0]
            if "/" in executable:
                available = os.access(ROOT / executable, os.X_OK)
            else:
                available = (
                    shutil.which(executable, path=model.get("env", {}).get("PATH"))
                    is not None
                )
            if not available:
                raise ValueError(
                    f"{model['id']}: executable not found: {executable}; build/prepare before reserving GPUs"
                )
    return suite


def aggregate(paths):
    summaries = [json.loads((path / "summary.json").read_text()) for path in paths]
    if len(summaries) != 2 or any(s["failed_requests"] for s in summaries):
        raise ValueError("need two complete, error-free measured passes")
    metrics = {}
    for key in ("successful_latency_p50_ms", "successful_latency_p95_ms"):
        values = [s[key] for s in summaries]
        if any(not bench.number(v) or v <= 0 for v in values):
            raise ValueError("invalid latency summary")
        metrics[key] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "passes": values,
        }
    for key in ("choice_accuracy", "noul_accuracy", "score_mae"):
        values = [s["quality_on_successful_requests"][key] for s in summaries]
        if values[0]["count"] != values[1]["count"]:
            raise ValueError("quality denominators differ")
        metrics[key] = {
            "count_per_pass": values[0]["count"],
            "passes": [v["value"] for v in values],
        }
    return metrics


def regression(reference, candidate, gates):
    errors = []
    paths = [path / f"measured-{i}" for path in (reference, candidate) for i in (1, 2)]
    configs = [json.loads((path / "config.json").read_text()) for path in paths]
    fixed = (
        "manifest_sha256",
        "validation_protocol",
        "score_decimals",
        "model",
        "protocol",
        "concurrency",
        "warmup",
        "timeout_seconds",
        "python",
        "httpx",
    )
    for config in configs[1:]:
        if any(config.get(k) != configs[0].get(k) for k in fixed) or any(
            config["metadata"].get(k) != configs[0]["metadata"].get(k)
            for k in CONTROL_KEYS
        ):
            raise ValueError(
                "baseline controls differ; establish a new baseline explicitly"
            )
    old, new = aggregate(paths[:2]), aggregate(paths[2:])
    for key in ("choice_accuracy", "noul_accuracy", "score_mae"):
        if old[key]["count_per_pass"] != new[key]["count_per_pass"]:
            raise ValueError("baseline quality denominator differs")
        if not old[key]["count_per_pass"]:
            continue
        if key == "score_mae":
            failed = (
                max(new[key]["passes"])
                > min(old[key]["passes"]) + gates["max_score_mae_increase"]
            )
        else:
            failed = (
                min(new[key]["passes"])
                < max(old[key]["passes"]) - gates["max_accuracy_drop"]
            )
        if failed:
            errors.append(f"{key} regression")
    for key in ("successful_latency_p50_ms", "successful_latency_p95_ms"):
        # Every measured pass must meet the baseline mean plus declared allowance.
        if new[key]["max"] > old[key]["mean"] * (1 + gates["max_latency_increase"]):
            errors.append(f"{key} regression")
    parity = []
    for path in paths[2:]:
        args = argparse.Namespace(
            reference=paths[0],
            candidate=path,
            max_probability_drift=gates["max_probability_drift"],
            max_score_drift=gates["max_score_drift"],
            max_flips=gates["max_flips"],
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            failed = bench.compare(args)
        parity.append(json.loads(output.getvalue()))
        if failed:
            errors.append(f"{path.name}: output parity failed")
    return {
        "passed": not errors,
        "errors": errors,
        "baseline": old,
        "candidate": new,
        "parity": parity,
    }


async def wait_ready(url, processes, timeout):
    started = time.monotonic()
    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
        while time.monotonic() - started < timeout:
            if any(p.poll() is not None for p in processes):
                raise ValueError("a service exited during startup; see service logs")
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return time.monotonic() - started
            except httpx.RequestError:
                pass
            await asyncio.sleep(0.5)
    raise ValueError("readiness timeout")


def stop_services(processes):
    # These process groups were created by this runner, including child GPU workers.
    for process in reversed(processes):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


async def run_model(model, suite, output, baseline, runtime):
    output.mkdir()
    processes, logs = [], []
    try:
        manifest = output / "manifest.jsonl"
        manifest.write_bytes((ROOT / model["manifest"]).read_bytes())
        cases = bench.load_cases(manifest)
        endpoints = {model["health"]}
        endpoints.update(model.get("readiness", []))
        if backend := model.get("env", {}).get("OMNI_JEV_BACKEND_URL"):
            endpoints.add(backend.rstrip("/") + "/health")
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            for endpoint in endpoints:
                try:
                    await client.get(endpoint)
                except httpx.RequestError:
                    pass
                else:
                    raise ValueError(
                        "health endpoint already occupied; use dedicated ports"
                    )
        env = {**os.environ, **model.get("env", {})}
        started = time.monotonic()
        for index, command in enumerate(model["commands"]):
            log = (output / f"service-{index}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        for endpoint in [model["health"], *model.get("readiness", [])]:
            remaining = suite.get("startup_timeout", 600) - (time.monotonic() - started)
            await wait_ready(endpoint, processes, remaining)
        readiness = time.monotonic() - started
        bench.write_json(
            output / "startup.json",
            {
                "seconds_to_health": readiness,
                "process_ids": [p.pid for p in processes],
                "commands": model["commands"],
            },
        )
        metadata = {
            **model["metadata"],
            "host": platform.node(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "gpu_uuids": runtime["gpu_uuids"],
            "harness_sha256": runtime["harness_sha256"],
            "runtime_revision": runtime["sha"],
            "reservation": {
                "scheduler": "gpu run",
                "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            },
            "cuda_evidence": {
                "operator_evidence": model["metadata"].get("cuda_evidence"),
                "service_logs": [f"service-{i}.log" for i in range(len(logs))],
                "scheduler_visibility": os.environ["CUDA_VISIBLE_DEVICES"],
            },
        }
        bench.write_json(output / "metadata.json", metadata)
        for phase in ("feasibility", "measured-1", "measured-2"):
            print(f"{model['id']}: {phase}", flush=True)
            args = argparse.Namespace(
                metadata=output / "metadata.json",
                output=output / phase,
                manifest=manifest,
                endpoint=model["endpoint"],
                model=model["model"],
                protocol=model.get("protocol", "systemone"),
                score_decimals=model.get("score_decimals", 4),
                phase="feasibility" if phase == "feasibility" else "measured",
                concurrency=suite.get("concurrency", 1),
                warmup=suite.get("warmup", 3),
                timeout=suite.get("request_timeout", 180),
            )
            if await bench.run(args, cases):
                raise ValueError(
                    f"{phase} failed; stopping this model at declared run limit"
                )
            if any(process.poll() is not None for process in processes):
                raise ValueError("a service exited during replay; see service logs")
        result = {
            "status": "baseline_recorded",
            "measurements_complete": True,
            "metrics": aggregate([output / f"measured-{i}" for i in (1, 2)]),
            "seconds_including_startup": time.monotonic() - started,
        }
        if baseline:
            comparison = regression(baseline / model["id"], output, suite["gates"])
            bench.write_json(output / "regression.json", comparison)
            result.update(
                status="passed" if comparison["passed"] else "failed",
                comparison=comparison,
            )
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "failed", "error": str(error)}
    finally:
        stop_services(processes)
        for log in logs:
            log.close()


def check_references(suite, output, report):
    for model in suite["models"]:
        reference = model.get("reference_model")
        result = report["models"][model["id"]]
        if not reference or not result.get("measurements_complete"):
            continue
        comparisons = []
        try:
            if not report["models"].get(reference, {}).get("measurements_complete"):
                raise ValueError(f"reference {reference} did not complete")
            for index in (1, 2):
                args = argparse.Namespace(
                    reference=output / reference / f"measured-{index}",
                    candidate=output / model["id"] / f"measured-{index}",
                    max_probability_drift=suite["gates"]["max_probability_drift"],
                    max_score_drift=suite["gates"]["max_score_drift"],
                    max_flips=suite["gates"]["max_flips"],
                )
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    bench.compare(args)
                comparisons.append(json.loads(captured.getvalue()))
            passed = all(comparison["passed"] for comparison in comparisons)
            result["reference_parity"] = {
                "reference_model": reference,
                "passed": passed,
                "passes": comparisons,
            }
            if not passed:
                result.update(
                    status="failed", error=f"output parity against {reference} failed"
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status="failed", error=str(error))
        bench.write_json(
            output / model["id"] / "reference-parity.json",
            result.get("reference_parity")
            or {"passed": False, "error": result.get("error")},
        )


async def run_suite(args, suite):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("GPU scheduler did not provide device visibility")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = {
        "sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    }
    runtime["harness_sha256"] = {}
    for name in (
        "benchmarks/weekly.py",
        "benchmarks/bench.py",
        "tests/e2e/clm_uncached_worker.py",
    ):
        source = HARNESS_ROOT / name
        runtime["harness_sha256"][name] = bench.digest(source)
        destination = args.output / "harness" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (args.output / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
    )
    # Include added source files too: git diff alone omits untracked work.
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    runtime["source_sha256"] = {
        name: bench.digest(ROOT / name) for name in files if (ROOT / name).is_file()
    }
    bench.write_json(args.output / "runtime.json", runtime)
    bench.write_json(args.output / "suite.json", suite)
    bench.write_json(
        args.output / "protocol.json",
        {
            "hypothesis": "The tested runtime preserves labelled accuracy, output fidelity and warm HTTP latency for the frozen workloads.",
            "isolated_variable": "runtime source revision/patch versus the explicitly chosen baseline; first run establishes a baseline candidate",
            "fixed_controls": list(CONTROL_KEYS)
            + [
                "manifest_sha256",
                "model",
                "protocol",
                "concurrency",
                "warmup",
                "timeout_seconds",
                "python",
                "httpx",
            ],
            "success_criterion": "Every selected path completes without errors and meets every declared suite gate against its baseline.",
            "stop_condition": "Stop a model on startup, warmup or request failure; do not retry or add passes.",
            "run_budget_per_model": {"feasibility_passes": 1, "measured_passes": 2},
        },
    )
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + args.gpu_ids,
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    )
    (args.output / "hardware.csv").write_text(hardware)
    devices = [
        [field.strip() for field in row.split(",")] for row in hardware.splitlines()
    ]
    runtime["gpu_uuids"] = [row[1] for row in devices]
    for model in suite["models"]:
        if model.get("unavailable"):
            continue
        metadata = model["metadata"]
        if metadata["gpu_ids"] != [int(row[0]) for row in devices] or any(
            metadata["gpu"] != row[2] or metadata["driver"] != row[3] for row in devices
        ):
            raise ValueError(f"{model['id']}: metadata differs from reserved hardware")
    bench.write_json(args.output / "runtime.json", runtime)
    report = {"baseline": str(args.baseline) if args.baseline else None, "models": {}}
    for model in suite["models"]:
        if model.get("unavailable"):
            result = {"status": "unavailable", "reason": model["unavailable"]}
        else:
            result = await run_model(
                model, suite, args.output / model["id"], args.baseline, runtime
            )
        report["models"][model["id"]] = result
        bench.write_json(args.output / "report.json", report)
    check_references(suite, args.output, report)
    statuses = [r["status"] for r in report["models"].values()]
    report["passed"] = bool(statuses) and all(s == "passed" for s in statuses)
    bench.write_json(args.output / "report.json", report)
    lines = [
        "# Weekly local E2E results",
        "",
        "| Model/path | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for name, result in report["models"].items():
        detail = result.get(
            "reason", result.get("error", "See per-model metrics and raw passes")
        )
        lines.append(f"| {name} | {result['status']} | {detail.replace('|', '/')} |")
    lines.extend(
        [
            "",
            "Measured pass variation (warm HTTP latency):",
            "",
            "| Model/path | Choice accuracy | Noul accuracy | Score MAE | p50 ms range | p95 ms range |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name, result in report["models"].items():
        metrics = result.get("metrics")
        if not metrics:
            continue
        values = []
        for key in ("choice_accuracy", "noul_accuracy", "score_mae"):
            metric = metrics[key]
            values.append(
                f"{min(metric['passes']):.4f}–{max(metric['passes']):.4f} (n={metric['count_per_pass']}/pass)"
                if metric["count_per_pass"]
                else "N/A"
            )
        for key in ("successful_latency_p50_ms", "successful_latency_p95_ms"):
            values.append(f"{metrics[key]['min']:.2f}–{metrics[key]['max']:.2f}")
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Results: {args.output / 'report.md'}", flush=True)
    return 0 if report["passed"] else (1 if "failed" in statuses else 2)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "_run"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT,
        help="runtime checkout to test; manifests may live outside it",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--models", help="comma-separated model/path IDs; omitted means full coverage"
    )
    parser.add_argument("--gpu-ids")
    parser.add_argument("--wait", default="10m")
    parser.add_argument("--timeout", default="2h")
    args = parser.parse_args()
    ROOT = args.repo.resolve()
    try:
        suite = load_suite(args.suite)
        if args.models:
            selected = args.models.split(",")
            if not selected or set(selected) - {m["id"] for m in suite["models"]}:
                raise ValueError("unknown selected model/path")
            suite["excluded_models"] = [
                m["id"] for m in suite["models"] if m["id"] not in selected
            ]
            suite["models"] = [m for m in suite["models"] if m["id"] in selected]
        for model in suite["models"]:
            if (
                not model.get("unavailable")
                and model.get("reference_model")
                and model["reference_model"] not in {m["id"] for m in suite["models"]}
            ):
                raise ValueError(
                    f"include reference profile {model['reference_model']} in --models"
                )
        if args.command == "validate":
            print(
                f"Validated {len(suite['models'])} model/path entries without GPU access"
            )
            return 0
        if not args.output:
            parser.error("run requires --output")
        args.output = args.output.resolve()
        if args.output.exists():
            raise ValueError(
                "output directory already exists; results are never overwritten"
            )
        if args.baseline:
            args.baseline = args.baseline.resolve()
            report = json.loads((args.baseline / "report.json").read_text())
            for model in suite["models"]:
                if (
                    not model.get("unavailable")
                    and report["models"]
                    .get(model["id"], {})
                    .get("measurements_complete")
                    is not True
                ):
                    raise ValueError(f"no complete baseline for {model['id']}")
        if args.command == "run":
            if not args.gpu_ids or not all(
                part.isdigit() for part in args.gpu_ids.split(",")
            ):
                parser.error("run requires exact --gpu-ids, e.g. 0 or 0,1")
            for flags in ([], ["-v"], ["-vv"], ["--no-schedule"]):
                subprocess.run(["gpu", "status", *flags], check=True)
            command = [
                "gpu",
                "run",
                "--gpu-ids",
                args.gpu_ids,
                "--wait",
                args.wait,
                "--timeout",
                args.timeout,
                "--note",
                "system1-omni weekly accuracy and latency E2E",
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                "_run",
                "--suite",
                str(args.suite.resolve()),
                "--output",
                str(args.output),
                "--gpu-ids",
                args.gpu_ids,
                "--repo",
                str(ROOT),
            ]
            if args.baseline:
                command.extend(["--baseline", str(args.baseline)])
            if args.models:
                command.extend(["--models", args.models])
            return subprocess.run(command, cwd=ROOT, check=False).returncode
        return asyncio.run(run_suite(args, suite))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
