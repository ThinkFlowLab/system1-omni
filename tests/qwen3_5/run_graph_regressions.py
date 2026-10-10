#!/usr/bin/env python3
"""Run the real-GPU cases and require unconditional capture-failure diagnostics."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def diagnostics(output: str) -> dict:
    pattern = re.compile(
        r"CUDA Graph capture failed in (text|multimodal) mode; "
        r"using eager execution: ([^\r\n]*)",
    )
    matches = pattern.findall(output)
    counts = {mode: sum(m == mode for m, _ in matches) for mode in ("text", "multimodal")}
    valid = (
        len(matches) == 2
        and counts == {"text": 1, "multimodal": 1}
        and all("injected record failure" in error for _, error in matches)
    )
    return {
        "diagnostic_count": len(matches),
        "diagnostics_by_mode": counts,
        "failure_diagnostics": [{"mode": mode, "error": error} for mode, error in matches],
        "unconditional_diagnostics_passed": valid,
    }


def main() -> int:
    test_dir = Path(__file__).resolve().parent
    repo = test_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="full Cargo output")
    parser.add_argument("--result", type=Path, help="JSON result; defaults to LOG.json")
    args = parser.parse_args()
    log_path = args.log.resolve()
    result_path = (
        args.result.resolve() if args.result else log_path.with_name(log_path.name + ".json")
    )
    if log_path == result_path:
        parser.error("--log and --result must be different paths")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("CUA_S1_GRAPH_TRACE", None)
    command = [
        "cargo", "test", "--offline", "--release", "--locked",
        "-p", "omni-qwen3-5-native", "--lib", "model::graph_tests::", "--",
        "--ignored", "--nocapture", "--test-threads=1",
    ]
    with log_path.open("wb") as log:
        tested = subprocess.run(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    cargo_returncode = tested.returncode
    summary = diagnostics(log_path.read_bytes().decode("utf-8", errors="replace"))
    counts = re.findall(
        r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )
    test_counts = [tuple(map(int, count)) for count in counts]
    passed = (
        cargo_returncode == 0
        and test_counts == [(3, 0, 0)]
        and summary["unconditional_diagnostics_passed"]
    )
    result = {
        "passed": passed,
        "test_counts": test_counts,
        "command": command,
        "cargo_returncode": cargo_returncode,
        "graph_trace_unset": "CUA_S1_GRAPH_TRACE" not in env,
        "full_log": str(log_path),
        **summary,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
