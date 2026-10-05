# Token-target reuse comparison

Run 2026-09-29, Slurm job 175104 (COMPLETED 0:0), one NVIDIA L40S.
[Results table](../../RESULTS.md#token-target-reuse-2026-09-29).

The candidate reuses the target IDs already in the GPU input tensor instead of
constructing another device tensor for each candidate. Everything else, including
FP32 log-softmax, reduction order, model weights and batch layout, is unchanged.

- `raw.jsonl.gz`: all 320 samples (8 cells x 2 variants x 20), including order,
  timing, peak allocated memory, selected value and candidate-score hashes.
- `summary.json.gz`: p50/p95, source hashes, environment and exact-parity checks.
- `verify.json.gz`: full 17-case/six-batch verifier, cache isolation, question
  independence and near-tie checks on the candidate engine.
- `pytest.log`: all 26 candidate Python tests pass.
- `probe.py.gz`: exact executed comparison script, compressed without changes.

Both variants share one read-only model. Three warmups precede each cell, then
baseline/candidate order alternates on each of 20 paired samples. Both choices
also match an independent uncached oracle. All 160 pairs have exactly equal
candidate scores and selections. CUDA is synchronized around each timed call.
The script is intentionally a small comparison, not a general benchmark runner.
The 16-candidate `32` and `all` settings take the same single-batch path.

These are descriptive results from one session. In particular, the first cell's
large p95 change should not be read as a repeatable tail-latency improvement.
Six cells have identical peak allocation; the two 255-candidate/batch-32 cells
increase by 4,608 bytes (less than 0.00021%), with the target view still live.

The earlier attempt (175098) stopped on a missing batch-size entry in the probe;
its partial timings are excluded. The complete run above uses the corrected
script; no model code or tolerance was changed in response to that failure.

## Reproduce

Use the pinned environment and model/reference downloads in
[the recipe](../../README.md). Replay the recorded revisions in an isolated
checkout; later readiness and usage fixes are not part of this comparison.
From the repository root:

```sh
git worktree add --detach .local/lfm/token-replay a402606cd9ee6a8c129be3df51b1c923ec7bf324
cd .local/lfm/token-replay
mkdir -p .local/lfm/readiness/baseline .local/lfm/readiness/candidate
gzip -cd recipe/lfm2/results/l40s-token-copy-20260929/probe.py.gz > .local/lfm/readiness/probe.py
git show f46f4582f36b525bcaf9b6e202220e3ba032f829:src/models/lfm2/engine.py > .local/lfm/readiness/baseline/engine.py
git show a402606cd9ee6a8c129be3df51b1c923ec7bf324:src/models/lfm2/engine.py > .local/lfm/readiness/candidate/engine.py
python .local/lfm/readiness/probe.py /path/to/new-results
python -m pytest src/models/lfm2/tests
python recipe/lfm2/verify.py --reference-dir /path/to/pinned/reference --device cuda --output /path/to/new-verify.json
```

Choose a fresh result directory because the script appends raw samples.
Compare source hashes against `summary.json.gz` before interpreting a rerun.
The baseline is the first published PR commit; the candidate engine SHA-256 is
`98ddc31f1f38f76a504aaae83aa3f222474330394d58e71ebcf336caf8bb4106`.

```sh
cd recipe/lfm2/results/l40s-token-copy-20260929
sha256sum -c SHA256SUMS
gzip -cd summary.json.gz
```
