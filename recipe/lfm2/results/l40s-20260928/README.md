# Raw LFM2.5-350M results

One NVIDIA L40S, 2026-09-28. [Analysis and full tables](../../RESULTS.md).

- `bench.jsonl.gz`: all 2,560 internal-scorer measured samples (16 cells x 8 variants x 20).
- `worker.jsonl.gz`: all 360 public-worker-path measured samples (18 cells x 20).
- `*.manifest.json.gz` and `*.summary.json.gz`: environment/source hashes, workload definitions, p50/p95 and allocated/reserved peaks.
- `verify.json.gz`: final 17-case x 6-batch verification, hybrid-state isolation, API independence and 4 x 6 near-tie checks. Includes raw candidate scores and margins.
- `frontend.json.gz`: all 10 direct-worker/frontend comparisons, including exact response bytes.
- `python-tests.log`, `rust-tests.log`: required local-code checks executed on Slurm compute nodes.
- `environment.freeze.txt`, `python-version.txt`: original observed environment. The two Conda build-path entries are represented by their installed versions in [requirements.lock.txt](../../../../src/models/lfm2/requirements.lock.txt): packaging 26.3, pip 26.2.1.

Original JSON and JSONL bytes are preserved with gzip compression only. No timing, score, or metadata row was dropped. Original source paths are retained in the manifests, with file SHA-256 values identifying the actual executed code. `SHA256SUMS` checks the archived files.

```sh
cd recipe/lfm2/results/l40s-20260928
sha256sum -c SHA256SUMS
gzip -cd verify.json.gz | jq '.status, .checks, .sections.near_ties'
gzip -cd bench.summary.json.gz | jq '.summary.variants'
gzip -cd bench.jsonl.gz | head -n 1
```

Provenance: CPU checks completed in Slurm job 174759; the full performance and frontend run completed in 174771; final numerical verification and environment capture completed in 174785. Only one GPU was used at a time. The final verifier tightens the earlier report by comparing every pair of batch settings and testing near ties at all six batch settings. The engine and worker hashes are identical throughout.

The unchanged Rust frontend comes from upstream commit `30622438bbe824563a066a0525631ae6e1beaabb`, built with Rust 1.94.0 in the official bookworm image. Python source hashes in the reports identify the uncommitted implementation used for these runs.
