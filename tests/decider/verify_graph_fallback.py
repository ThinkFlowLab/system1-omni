#!/usr/bin/env python3
"""Prove genuine Decider warmup survives capture failure and stays eager."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from verify_reference import VALIDATION_PROTOCOL, compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "binary", "wrapper", "parity_output", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.parity_output / "protocol.json").read_text())
    reference = [json.loads(x) for x in (args.parity_output / "reference.jsonl").read_text().splitlines()]
    selected_env = {"DECIDER_GRAPH": "1", "DECIDER_PREFIX": "0", "DECIDER_FIXED": "0",
                    "DECIDER_BATCH_MAX_ROWS": "4", "DECIDER_BATCH_MAX_TOKENS": "4096", "CUA_S1_GRAPH": "1"}
    env = {**os.environ, **selected_env}
    # Preserve the selected path and revised gates even if startup or capture fails.
    (args.output / "protocol.json").write_text(json.dumps({**protocol, "validation_protocol": VALIDATION_PROTOCOL,
        "capture_fault": "cs1_graph_begin returns2", "env": selected_env}, indent=2) + "\n")
    requests = "".join(json.dumps(x["request"], ensure_ascii=False) + "\n" for x in protocol["cases"])
    with (args.output / "stderr.log").open("w") as stderr:
        result = subprocess.run([str(args.binary), str(args.model), str(args.wrapper)], env=env,
                                input=requests, stdout=subprocess.PIPE, stderr=stderr, text=True, check=True)
    (args.output / "native.jsonl").write_text(result.stdout)
    records = [json.loads(x) for x in result.stdout.splitlines()]
    assert len(records) == len(reference)
    for ref, got in zip(reference, records):
        compare(ref["response"], got["response"])
        stats = got["graph"]
        assert stats["requested"] and not stats["enabled"]
        assert stats["fallbacks"] == 1 and stats["captures"] == stats["replays"] == 0
    (args.output / "summary.json").write_text(json.dumps({"passed": len(records), "graph": records[-1]["graph"]}, indent=2))


if __name__ == "__main__":
    main()
