"""Measure warmed native CLI requests; run bundles in forward/reverse order."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def percentile(values, q):
    values = sorted(values)
    p = (len(values) - 1) * q
    lo = int(p)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (p - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/laya-run"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--original-rope", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1:
        parser.error("warmup and samples must be positive")
    fixtures = Path(__file__).resolve().parents[3] / "tests/laya/data/requests.json"
    cases = [c for c in json.loads(fixtures.read_text())
             if c["name"] in ["choice", "score", "short_1", "medium_1", "long_1"]]
    command = [str(args.binary.resolve()), str(args.checkpoint.resolve()), str(args.bundle.resolve())]
    command += [flag for flag, enabled in [("--eager", args.eager), ("--original-rope", args.original_rope)] if enabled]
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for key in ["LAYA_RAW_LOGITS", "LAYA_DUMP_DIR", "LAYA_DUMP_HIDDEN", "LAYA_VERIFY_WEIGHTS"]:
        env.pop(key, None)
    result = {"command": command, "binary_sha256": sha(args.binary),
              "library_sha256": sha(args.bundle / "liblaya_cuda.so"),
              "fixtures_sha256": sha(fixtures), "warmup": args.warmup, "samples": args.samples,
              "boundary": "CLI parse/pack/upload/forward/heads/readback/decode/JSON write; startup and HTTP excluded",
              "cases": {}}
    selector = None
    logs = {}
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env, bufsize=0)
    try:
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        pending = []
        for kind in buffers:
            logs[kind] = (args.output / f"{kind}.log").open("w")

        def event(deadline):
            while not pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("native response deadline expired")
                if not selector.get_map():
                    raise RuntimeError(f"native exited: {proc.poll()}")
                for key, _ in selector.select(remaining):
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    buffer = buffers[key.data]
                    buffer.extend(data)
                    while b"\n" in buffer:
                        line, _, rest = buffer.partition(b"\n")
                        buffer[:] = rest
                        pending.append((key.data, line.decode()))
            kind, line = pending.pop(0)
            logs[kind].write(line + "\n")
            return kind, line

        def infer(request):
            payload = memoryview((json.dumps(request, ensure_ascii=False) + "\n").encode())
            while payload:
                written = proc.stdin.write(payload)
                if not written:
                    raise BrokenPipeError("native stdin stopped accepting input")
                payload = payload[written:]
            deadline = time.monotonic() + 10
            response, elapsed = None, None
            while response is None or elapsed is None:
                kind, line = event(deadline)
                if kind == "stdout":
                    if response is not None:
                        raise RuntimeError("extra response")
                    response = json.loads(line)
                    if "error" in response:
                        raise RuntimeError(response)
                elif line.startswith("engine_wall_ms="):
                    elapsed = float(line.split("=", 1)[1])
                    if not math.isfinite(elapsed) or elapsed <= 0:
                        raise ValueError("invalid engine timing")
            return response, elapsed

        deadline = time.monotonic() + 120
        while event(deadline) != ("stderr", "READY native Laya (Rust + CUDA)"):
            pass
        maps = Path(f"/proc/{proc.pid}/maps").read_text()
        (args.output / "native.maps").write_text(maps)
        loaded = {line.split(maxsplit=5)[-1] for line in maps.splitlines() if len(line.split(maxsplit=5)) == 6}
        if str((args.bundle / "liblaya_cuda.so").resolve()) not in loaded:
            raise RuntimeError("requested CUDA bundle is not loaded")
        if "libpython" in maps or "libtorch" in maps:
            raise RuntimeError("unexpected Python/Torch runtime mapping")
        for case in cases:
            for _ in range(args.warmup):
                expected, _ = infer(case["request"])
            timings = []
            for _ in range(args.samples):
                response, ms = infer(case["request"])
                if response != expected:
                    raise ValueError("response changed between repetitions")
                timings.append(ms)
            result["cases"][case["name"]] = {
                "engine_wall_ms": timings, "p50_ms": percentile(timings, .5),
                "p95_ms": percentile(timings, .95), "response": expected,
            }
        proc.stdin.close()
        if proc.wait(timeout=10) != 0:
            raise RuntimeError("native exited unsuccessfully")
        (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if selector is not None:
            selector.close()
        for stream in logs.values():
            stream.close()


if __name__ == "__main__":
    main()
