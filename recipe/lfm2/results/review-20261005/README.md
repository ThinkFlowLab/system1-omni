# LFM2.5-350M review validation — 2026-10-05

This bundle records correctness and integration checks of source commit
`3e2cbcf9f6e6f2db05642739efd46f84b9db4d29` against upstream main
`f594d7dfc4c2bef812e23f7ed73573be9625b287`. The tested source archive
SHA-256 is `f32bc518e8e60e5fc66c67087d2234d7fed34a46fccfe865e6de32a235784b68`.
Both successful runs recorded 215 source files in their hash manifests.
[Provenance](provenance.json) gives job states, pins, and repository-relative source hashes.
These checks do not add performance measurements to the [historical results](../../RESULTS.md).

CPU retry job 184536 completed 0:0 on cpu-2 in 31 seconds. The LFM suite had
**32 passed**; the Rust workspace had **54 passed, 9 ignored**, with three pinned fixture tests
and six native CUDA kernel tests ignored; the explicit frontend command had **12 passed** (included in the workspace total). Rust fmt, Clippy and release
build passed. Strict MkDocs build, benchmark fixture validation and **8 benchmark
unit tests** passed. The raw batch output is [review-cpu-184536.out](review-cpu-184536.out);
[cpu-checks.tar.gz](cpu-checks.tar.gz) losslessly wraps the downloaded
`cpu-184536.tar`, which contains 18 check logs, environment/pin records and
hash records.

GPU job 184541 completed 0:0 on one NVIDIA L40S (driver 580.178.04)
in 54 seconds, using FP16
and pinned LiquidAI/LFM2.5-350M weights. [verify.json.gz](verify.json.gz)
preserves the full verifier JSON: **17 cases × 6 batch sizes = 102**
score/token-ID/selection checks passed against the uncached oracle; maximum
absolute score error **0.1276397705078125** was below the empirical **0.15**
gate. All six attention and ten convolution cache layers passed isolation and
distribution checks. Four near-tie probes passed at all six sizes (**24 checks**),
with minimum oracle margin **0.0462646484375** and no selection flips.
The multi-question API check passed independent answers and additive, unpadded
input usage (**193 + 162 = 355**). The direct worker and Rust frontend matched
in **10/10** cases, including health, valid responses and 400/422 errors;
the full byte-parity report is [frontend.json.gz](frontend.json.gz).

Initial CPU job 184529 was **FAILED 1:0** even though its individual checks
printed PASS and the binary hash was written. Its
[batch output](review-cpu-184529.out) and
[Rust step log](rust-checks-184529.log) are retained. The corrected retry used
a non-login shell wrapper and completed; the available logs do not uniquely
prove which exit hook or container step caused the first failure. No source
code or test gate was changed for this retry.

To reproduce, use the [LFM recipe](../../README.md) for the pinned environment,
weights, reference checkout and loopback worker/frontend setup. From the
repository root at the tested commit, the CPU checks are
`python -m pytest -q tests/lfm2`,
`cargo fmt --all --check`,
`cargo clippy --workspace --locked --all-targets -- -D warnings`,
`cargo test -p omni-jev --test frontend --locked`,
`cargo test --workspace --locked`, and
`cargo build --workspace --release --locked`. For the docs and benchmark fixture checks, use a separate tools environment
so the pinned LFM environment stays intact:

    python3.12 -m venv .venv-checks
    .venv-checks/bin/python -m pip install -r docs/requirements.txt -r benchmarks/requirements.txt
    .venv-checks/bin/python -m mkdocs build --strict
    .venv-checks/bin/python benchmarks/bench.py validate benchmarks/smoke.jsonl
    .venv-checks/bin/python -m unittest discover -s tests/benchmarks -p 'test_*.py' -v

On a Slurm GPU node,
run `python recipe/lfm2/verify.py --reference-dir <pinned-reference> --device cuda --output <verify.json>`; launch the worker with `--candidate-batch-size 8`
and compare its loopback endpoint with the frontend using
`recipe/lfm2/compare_with_frontend.py`. The exact run settings and source
hashes are in the reports and provenance file.

All three `.gz` files have gzip mtime zero and decompress to the exact
downloaded source bytes. Run `sha256sum -c SHA256SUMS` here to verify this
bundle. The score tolerance is a gate for these cases, not a uniform FP16
guarantee. This run did not repeat latency or memory benchmarks.
