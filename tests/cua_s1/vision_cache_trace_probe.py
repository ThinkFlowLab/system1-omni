#!/usr/bin/env python3
"""Check live A/B/A CUDA logs after the Rust feature-parity probe.

Run vision_cache_aba_probe with CUA_S1_VISION_CACHE_ENTRIES=4 and
CUA_S1_GRAPH_TRACE=1, redirect both stdout and stderr, then pass that log here.
The baseline single-grid implementation must fail this assertion.
"""
import pathlib
import sys

log = pathlib.Path(sys.argv[1]).read_text()
assert "test result: ok. 1 passed" in log, "the GPU feature-parity probe must pass"
assert log.count("VISION_CACHE_ABA_BEGIN") == log.count("VISION_CACHE_ABA_END") == 1
window = log.split("VISION_CACHE_ABA_BEGIN")[1].split("VISION_CACHE_ABA_END")[0]
captures = window.count("Vision CUDA Graph captured")
replays = window.count("Vision CUDA Graph replayed")
assert (captures, replays) == (2, 1), (
    f"A/B/A expected 2 captures and 1 replay, observed {captures} captures and {replays} replays"
)
print(f"A/B/A retained graph: {captures} captures, {replays} replay; feature parity passed")
