# JEV-27B-VL validation

## Historical H800 validation

The ABI 6 candidate was measured on one H800 80 GB with CUDA 13.0.88,
driver 580.159.03 and BF16 language weights from
`autotrust/JEV-27B-VL@f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc`.
The frozen corpus has 36 synthetic text requests and 12 questions about one
synthetic image. It checks implementation parity, not general model quality.

Jobs 413587 and 413597 tested the same library and worker binaries. The measured
source snapshot predates comment, test-registration and diagnostic-header edits;
its hashes and those of the binaries are in the
[provenance record](https://github.com/linear3735/system1-omni/blob/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006/provenance.json).
These are historical measurements, not a new GPU test of later revisions.

- 12 GPU/kernel and checkpoint checks passed.
- Cache off/on each matched 48/48 reference decisions. Maximum probability
  differences were 0.021545 / 0.014215, within the fixed 0.025 tolerance.
- Worker/frontend replay passed 192 measured decisions. Eight worker and seven
  supported frontend error probes passed. The frontend owns unsupported-route
  404 responses; the original failed comparison is retained in the archive.

Each cache mode ran two passes of the same 12 image questions on the same GPU
and binary, at concurrency 1, with 12 excluded warmups before each pass:

| Mode | Pass 1 p50 | Pass 2 p50 | Combined p50 |
| --- | ---: | ---: | ---: |
| Cache off | 102.35 ms | 104.32 ms | 102.88 ms |
| Cache hit | 37.32 ms | 36.78 ms | 37.04 ms |

The combined-median ratio is **2.78×**. Timing includes worker-direct localhost
HTTP and response JSON parsing. It excludes startup, offline vision encoding,
warmup and the frontend. Peak memory and multi-client performance were not
measured. Raw timings, cache counters, verdicts and the initial failed comparison
are in the [fixed result archive](https://github.com/linear3735/system1-omni/tree/522f2256876a62ddd70873ee9e775055416aea1b/recipe/jev_vl/evidence/review-20261006).
[Earlier measurements and their limitations](https://github.com/linear3735/system1-omni/blob/4927d2a3373ae3e2d825bf63a320178988540031/recipe/jev_vl/validation.md#appendix-historical-evidence-audited-on-2026-10-06)
remain archived separately.

## Restore the frozen corpus

The maintained image, manifest template, restoration helper and reference
responses are in `tests/jev_vl/data/replay/`. From the repository root, restore
the manifest into a temporary directory:

```sh
jev_vl_evidence=$(mktemp -d "${TMPDIR:-/tmp}/jev-vl-evidence.XXXXXX")
python3 tests/jev_vl/data/replay/restore_manifest.py \
  --out "$jev_vl_evidence/manifest.jsonl"
```

The restore script verifies the original manifest SHA-256
`f72d1beaaaf53933d0a6edda26b635f46931990d8cdb56ca5c6a7ca94d2eb0ee`.
It restores approximately 41 MiB of inputs. The fixture's `reference/sha256.json`
records hashes of the 48 reference answers and eight error probes.
These inputs and responses do not establish the provenance of a new executable.

## Replay

Follow the [deployment recipe](README.md) to export weights, build the worker,
preencode `"$jev_vl_evidence/manifest.jsonl"`, and start serving. Keep the shell
variable from the restoration step. Use a new output directory for every run:

```sh
python3 recipe/jev_vl/replay.py \
  --base http://127.0.0.1:8001 --manifest "$jev_vl_evidence/manifest.jsonl" \
  --reference tests/jev_vl/data/replay/reference \
  --out "$jev_vl_evidence/worker-parity" --warmup 4 --passes 1
```

The replay tool returns nonzero for failed requests, changed decisions, invalid
probabilities or absolute probability drift above 0.025. Repeat through the
frontend at `http://127.0.0.1:8080` with a different output directory. Archived
error probes are separate from this replay; validate error handling separately.

For a paired cache measurement, start the worker with `JEV_VL_CACHE=0`, then run:

```sh
python3 recipe/jev_vl/replay.py \
  --base http://127.0.0.1:8001 --manifest "$jev_vl_evidence/manifest.jsonl" \
  --reference tests/jev_vl/data/replay/reference --pattern img- \
  --out "$jev_vl_evidence/cache-off" --warmup 12 --passes 2
```

Restart the same binary with `JEV_VL_CACHE=1` and repeat into `cache-on`. Keep the
GPU, checkpoint, prepared image assets and request order fixed. Save
`GET /v1/cache/stats` before and after each run to check cache use; counters
include warmup. Keep generated outputs outside the source tree.

## Coverage limits

This corpus does not cover general model quality, changed-image workloads,
cache eviction, multi-client performance or full Cua-S1/Open-Jev checkpoint
regressions. Clean installation remains unverified. Use the
[PR checks and review](https://github.com/ThinkFlowLab/system1-omni/pull/96)
for revision-specific CI and review status; passing this corpus does not replace
request-validation tests or a review of later code changes.
