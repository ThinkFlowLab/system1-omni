# LFM2.5-350M execution results

A later [token-target reuse comparison](#token-target-reuse-2026-09-29) records the small follow-up optimization separately.

The tables below describe the recorded September revisions. Those workers
reported prefix-only usage. The October review fix adds candidate input tokens
and startup warmup; the original measurements and raw records are retained.
Current validation uses `tests/lfm2` and the updated verifier.

Baseline run (published commit f46f458): 2026-09-28, one NVIDIA L40S (48 GiB), FP16; Python 3.12.14, PyTorch 2.14.0+cu130, Transformers 5.17.0, CUDA 13.0, driver 595.71.05. Batch size is explicit. These are inference/execution measurements on unchanged 350M weights.

All runs completed successfully. Raw unmodified records, manifests, environment freeze and check logs are in [the result bundle](results/l40s-20260928/README.md). The JSON/JSONL files are losslessly compressed; no measurements were removed.

## Correctness

The final GPU check covers 17 fixed diagnostic/stress inputs × batch sizes 1/8/16/32/64/all. All 102 actual forward token-ID checks, score tolerances and selected candidates match the independent uncached oracle. Maximum absolute candidate-score error is 0.12763977 against the fixed reference test gate of 0.15. The gate is empirical on these inputs and is not a uniform FP16 error guarantee.

All six attention and ten convolution layers preserve the common cache. Cached/uncached token distributions pass atol=0.015, rtol=0.08; token argmax matches. Question addition, reordering and ID renaming leave answers unchanged. Worker/frontend status, Content-Type and body bytes match in 10/10 cases, including 400/422 errors. CPU tests: 26 passed; repository Rust fmt, Clippy, tests and release build passed.

All four ambiguous near-tie probes pass at all six batch sizes (24 checks), with no observed selection flip. Minimum oracle margin: 0.04626465. Maximum score difference between any two batch settings in the 17-case suite: 0.09712601. Raw near-tie scores and margins are preserved per batch.

For the 255-candidate, long-context, mixed-length cell (3,877 prefix tokens), batch 32 reduces peak allocated memory from **17.97 GiB to 2.88 GiB** (84.0%) while p50 increases from **328.84 ms to 357.25 ms** (8.6%). All-at-once remains faster. This is a measured trade-off on one workload, not a default batch recommendation.

For two candidates and short context, the uncached baseline is faster: 18.3–18.6 ms p50 versus 21.3–21.4 ms for all-at-once branching. Sharing a prefix is not always a latency win.

## Independent public questions

The real Worker.systemone path includes request/response JSON encoding, with no HTTP transport. Every question gets its own prefill. Each cell has 16 candidates, 3 warmups and 20 measured samples. All 360 measured requests preserve choices and probabilities against their configuration's warmup output; usage is the sum of per-question prompt tokens.

Latency shown as p50 / p95, in milliseconds. `all` is 16 simultaneous branches per question.

| State | Questions | Batch 1 | Batch 8 | All |
| --- | ---: | ---: | ---: | ---: |
| Short | 1 | 168.7 / 186.7 | 32.3 / 35.6 | 22.4 / 25.0 |
| Short | 3 | 505.0 / 529.1 | 97.5 / 104.6 | 67.4 / 72.4 |
| Short | 8 | 1345.0 / 1372.9 | 261.3 / 264.3 | 179.1 / 183.7 |
| Long | 1 | 187.8 / 192.4 | 52.2 / 54.2 | 42.7 / 44.3 |
| Long | 3 | 560.7 / 571.8 | 155.7 / 157.7 | 127.5 / 129.2 |
| Long | 8 | 1501.1 / 1529.9 | 416.4 / 424.3 | 340.9 / 345.4 |

Peak allocated memory is the same for 1, 3 and 8 questions in these cells: short state uses 751.4 / 781.6 / 868.5 MiB at batch 1 / 8 / all. Long state uses 1286.2 MiB at every batch size because prefill dominates the peak. Peak includes resident weights and temporary activations. Reserved memory is allocator history and is recorded separately.

## Interpretation and limits

- Full branching is fastest in the 16-candidate public-worker cells. Smaller batches exchange latency for lower branch memory; prefill can still set the overall peak. There is no universal optimum or hard byte-budget guarantee.
- The model uses eager attention and the PyTorch convolution fallback. Every variant uses the same implementation; these are not native CUDA kernel measurements.
- The internal scorer matrix and public worker table have different schemas and prompts. Multi-field reference results are correctness regressions, not public multi-question performance.
- The uncached oracle projects all input positions to vocabulary logits. It is an intentionally simple correctness baseline, not a tuned uncached server. Any speedup over it also includes avoided output-projection work. Full versus bounded branching is the direct batching comparison.
- Candidate-relative scores and entropy concentration are uncalibrated, sensitive to length, wording and tokenization. Gold labels are hand-authored diagnostics, not a model-quality benchmark.
- Results come from one GPU/session and concurrency one. Twenty warm samples per cell provide descriptive percentiles, not confidence intervals or broad hardware conclusions.

## Full internal scorer matrix

Every cell uses 3 warmups and 20 measured samples for each of eight variants: 2,560 samples total, all selections matching the oracle, no OOM. Prefix lengths include the full schema and chat template. Candidate labels are `code_NNN` (short) or alternate with `long_code_value_NNN_with_extra_padding` (mixed). Context construction is fixed in `bench.py`.

All memory values below are peak allocated MiB, including weights. Latencies are milliseconds. Configured batch sizes above the candidate count execute the same single-batch path; small differences between those rows reflect separate samples.

### 2 candidates

| State / value length | Prefix tokens | Variant | p50 ms | p95 ms | Peak MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| short / short | 244 | uncached_oracle | 18.59 | 20.88 | 722.3 |
| short / short | 244 | reference_all | 20.60 | 22.03 | 703.5 |
| short / short | 244 | engine_b1 | 30.78 | 33.04 | 701.3 |
| short / short | 244 | engine_b8 | 21.03 | 22.58 | 703.5 |
| short / short | 244 | engine_b16 | 21.11 | 23.20 | 703.5 |
| short / short | 244 | engine_b32 | 20.80 | 22.52 | 703.5 |
| short / short | 244 | engine_b64 | 21.11 | 22.34 | 703.5 |
| short / short | 244 | engine_all | 21.42 | 22.60 | 703.5 |
| short / mixed | 251 | uncached_oracle | 18.28 | 21.31 | 724.6 |
| short / mixed | 251 | reference_all | 20.66 | 22.56 | 708.7 |
| short / mixed | 251 | engine_b1 | 31.11 | 33.41 | 701.5 |
| short / mixed | 251 | engine_b8 | 21.02 | 23.31 | 708.7 |
| short / mixed | 251 | engine_b16 | 21.17 | 22.96 | 708.7 |
| short / mixed | 251 | engine_b32 | 21.30 | 23.04 | 708.7 |
| short / mixed | 251 | engine_b64 | 21.10 | 23.22 | 708.7 |
| short / mixed | 251 | engine_all | 21.32 | 23.05 | 708.7 |
| long / short | 1716 | uncached_oracle | 42.34 | 43.67 | 1082.0 |
| long / short | 1716 | reference_all | 31.24 | 34.09 | 1092.7 |
| long / short | 1716 | engine_b1 | 41.91 | 44.65 | 1092.7 |
| long / short | 1716 | engine_b8 | 31.94 | 33.98 | 1092.7 |
| long / short | 1716 | engine_b16 | 31.99 | 32.98 | 1092.7 |
| long / short | 1716 | engine_b32 | 31.72 | 34.25 | 1092.7 |
| long / short | 1716 | engine_b64 | 31.93 | 34.39 | 1092.7 |
| long / short | 1716 | engine_all | 31.95 | 34.01 | 1092.7 |
| long / mixed | 1723 | uncached_oracle | 42.45 | 43.16 | 1088.2 |
| long / mixed | 1723 | reference_all | 31.53 | 33.37 | 1095.8 |
| long / mixed | 1723 | engine_b1 | 42.10 | 44.07 | 1095.8 |
| long / mixed | 1723 | engine_b8 | 32.04 | 34.72 | 1095.8 |
| long / mixed | 1723 | engine_b16 | 32.02 | 33.52 | 1095.8 |
| long / mixed | 1723 | engine_b32 | 31.84 | 33.15 | 1095.8 |
| long / mixed | 1723 | engine_b64 | 32.04 | 33.83 | 1095.8 |
| long / mixed | 1723 | engine_all | 31.92 | 34.15 | 1095.8 |

### 16 candidates

| State / value length | Prefix tokens | Variant | p50 ms | p95 ms | Peak MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| short / short | 314 | uncached_oracle | 136.93 | 141.02 | 730.8 |
| short / short | 314 | reference_all | 22.00 | 24.49 | 780.7 |
| short / short | 314 | engine_b1 | 167.50 | 178.19 | 706.0 |
| short / short | 314 | engine_b8 | 32.92 | 34.57 | 737.1 |
| short / short | 314 | engine_b16 | 22.58 | 24.21 | 780.7 |
| short / short | 314 | engine_b32 | 22.64 | 25.99 | 780.7 |
| short / short | 314 | engine_b64 | 22.50 | 25.00 | 780.7 |
| short / short | 314 | engine_all | 22.57 | 24.92 | 780.7 |
| short / mixed | 370 | uncached_oracle | 138.91 | 150.07 | 739.6 |
| short / mixed | 370 | reference_all | 22.54 | 23.44 | 806.2 |
| short / mixed | 370 | engine_b1 | 167.75 | 173.61 | 713.7 |
| short / mixed | 370 | engine_b8 | 32.66 | 34.59 | 752.4 |
| short / mixed | 370 | engine_b16 | 22.72 | 24.20 | 806.2 |
| short / mixed | 370 | engine_b32 | 22.62 | 24.38 | 806.2 |
| short / mixed | 370 | engine_b64 | 22.82 | 24.23 | 806.2 |
| short / mixed | 370 | engine_all | 23.05 | 23.99 | 806.2 |
| long / short | 1786 | uncached_oracle | 334.94 | 335.85 | 1113.4 |
| long / short | 1786 | reference_all | 34.66 | 35.90 | 1195.4 |
| long / short | 1786 | engine_b1 | 180.29 | 189.29 | 1125.2 |
| long / short | 1786 | engine_b8 | 45.00 | 46.73 | 1125.2 |
| long / short | 1786 | engine_b16 | 34.92 | 36.40 | 1195.4 |
| long / short | 1786 | engine_b32 | 34.65 | 35.76 | 1195.4 |
| long / short | 1786 | engine_b64 | 34.70 | 36.48 | 1195.4 |
| long / short | 1786 | engine_all | 34.99 | 36.33 | 1195.4 |
| long / mixed | 1842 | uncached_oracle | 346.88 | 348.02 | 1142.8 |
| long / mixed | 1842 | reference_all | 36.33 | 36.91 | 1239.0 |
| long / mixed | 1842 | engine_b1 | 181.85 | 190.40 | 1151.0 |
| long / mixed | 1842 | engine_b8 | 46.23 | 48.12 | 1151.0 |
| long / mixed | 1842 | engine_b16 | 36.04 | 37.14 | 1239.0 |
| long / mixed | 1842 | engine_b32 | 36.01 | 37.29 | 1239.0 |
| long / mixed | 1842 | engine_b64 | 36.02 | 37.30 | 1239.0 |
| long / mixed | 1842 | engine_all | 36.72 | 37.41 | 1239.0 |

### 64 candidates

| State / value length | Prefix tokens | Variant | p50 ms | p95 ms | Peak MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| short / short | 554 | uncached_oracle | 547.72 | 585.15 | 760.8 |
| short / short | 554 | reference_all | 29.06 | 30.30 | 1312.9 |
| short / short | 554 | engine_b1 | 632.08 | 688.05 | 740.2 |
| short / short | 554 | engine_b8 | 96.79 | 102.03 | 775.6 |
| short / short | 554 | engine_b16 | 56.80 | 61.82 | 847.9 |
| short / short | 554 | engine_b32 | 36.86 | 39.11 | 1002.6 |
| short / short | 554 | engine_b64 | 28.75 | 29.80 | 1312.9 |
| short / short | 554 | engine_all | 29.29 | 31.06 | 1312.9 |
| short / mixed | 778 | uncached_oracle | 554.74 | 578.09 | 790.6 |
| short / mixed | 778 | reference_all | 35.77 | 37.19 | 1613.1 |
| short / mixed | 778 | engine_b1 | 636.81 | 678.85 | 781.6 |
| short / mixed | 778 | engine_b8 | 99.03 | 102.77 | 810.8 |
| short / mixed | 778 | engine_b16 | 58.19 | 60.68 | 924.7 |
| short / mixed | 778 | engine_b32 | 38.64 | 42.73 | 1154.8 |
| short / mixed | 778 | engine_b64 | 34.86 | 36.62 | 1613.1 |
| short / mixed | 778 | engine_all | 35.50 | 36.84 | 1613.1 |
| long / short | 2026 | uncached_oracle | 1643.40 | 1656.95 | 1230.6 |
| long / short | 2026 | reference_all | 62.45 | 63.50 | 2919.0 |
| long / short | 2026 | engine_b1 | 649.09 | 663.88 | 1243.3 |
| long / short | 2026 | engine_b8 | 114.82 | 118.83 | 1243.3 |
| long / short | 2026 | engine_b16 | 77.03 | 79.16 | 1262.2 |
| long / short | 2026 | engine_b32 | 65.82 | 68.66 | 1814.8 |
| long / short | 2026 | engine_b64 | 62.55 | 63.87 | 2919.0 |
| long / short | 2026 | engine_all | 63.21 | 64.55 | 2919.0 |
| long / mixed | 2250 | uncached_oracle | 2075.76 | 2085.40 | 1359.9 |
| long / mixed | 2250 | reference_all | 80.04 | 82.21 | 3304.1 |
| long / mixed | 2250 | engine_b1 | 653.00 | 672.69 | 1371.7 |
| long / mixed | 2250 | engine_b8 | 122.83 | 125.89 | 1371.7 |
| long / mixed | 2250 | engine_b16 | 88.75 | 90.53 | 1371.7 |
| long / mixed | 2250 | engine_b32 | 81.94 | 82.70 | 2010.0 |
| long / mixed | 2250 | engine_b64 | 79.71 | 80.61 | 3304.1 |
| long / mixed | 2250 | engine_all | 79.93 | 81.48 | 3304.1 |

### 255 candidates

| State / value length | Prefix tokens | Variant | p50 ms | p95 ms | Peak MiB |
| --- | ---: | --- | ---: | ---: | ---: |
| short / short | 1509 | uncached_oracle | 4252.66 | 4260.48 | 997.2 |
| short / short | 1509 | reference_all | 122.98 | 125.16 | 7278.3 |
| short / short | 1509 | engine_b1 | 2479.54 | 2535.58 | 1008.5 |
| short / short | 1509 | engine_b8 | 360.94 | 369.53 | 1008.5 |
| short / short | 1509 | engine_b16 | 203.78 | 210.30 | 1120.6 |
| short / short | 1509 | engine_b32 | 147.65 | 149.95 | 1532.0 |
| short / short | 1509 | engine_b64 | 131.12 | 135.12 | 2356.9 |
| short / short | 1509 | engine_all | 123.21 | 126.09 | 7278.3 |
| short / mixed | 2405 | uncached_oracle | 9155.87 | 9174.87 | 1450.6 |
| short / mixed | 2405 | reference_all | 211.51 | 212.89 | 11728.1 |
| short / mixed | 2405 | engine_b1 | 2510.53 | 2574.80 | 1463.2 |
| short / mixed | 2405 | engine_b8 | 393.46 | 409.67 | 1463.2 |
| short / mixed | 2405 | engine_b16 | 259.16 | 263.28 | 1463.2 |
| short / mixed | 2405 | engine_b32 | 231.87 | 236.16 | 2099.6 |
| short / mixed | 2405 | engine_b64 | 225.87 | 228.37 | 3481.1 |
| short / mixed | 2405 | engine_all | 211.90 | 214.26 | 11728.1 |
| long / short | 2981 | uncached_oracle | 12004.16 | 12039.50 | 1839.9 |
| long / short | 2981 | reference_all | 229.17 | 232.32 | 13631.8 |
| long / short | 2981 | engine_b1 | 2509.43 | 2570.79 | 1859.7 |
| long / short | 2981 | engine_b8 | 405.42 | 423.18 | 1859.7 |
| long / short | 2981 | engine_b16 | 286.28 | 291.04 | 1859.7 |
| long / short | 2981 | engine_b32 | 252.22 | 255.07 | 2341.1 |
| long / short | 2981 | engine_b64 | 240.60 | 244.54 | 3964.0 |
| long / short | 2981 | engine_all | 229.26 | 232.84 | 13631.8 |
| long / mixed | 3877 | uncached_oracle | 18065.76 | 18100.31 | 2622.9 |
| long / mixed | 3877 | reference_all | 327.93 | 332.07 | 18400.4 |
| long / mixed | 3877 | engine_b1 | 2580.20 | 2719.31 | 2642.2 |
| long / mixed | 3877 | engine_b8 | 456.81 | 474.73 | 2642.2 |
| long / mixed | 3877 | engine_b16 | 372.67 | 382.01 | 2642.2 |
| long / mixed | 3877 | engine_b32 | 357.25 | 363.59 | 2949.7 |
| long / mixed | 3877 | engine_b64 | 341.50 | 352.43 | 5166.7 |
| long / mixed | 3877 | engine_all | 328.84 | 333.69 | 18400.4 |

## Token-target reuse (2026-09-29)

The follow-up reuses candidate token IDs already on the GPU for the score gather.
Against published commit f46f458, only the target-tensor construction changes.
On one L40S, all 160 paired comparisons have exactly equal scores and selections;
the 26 Python tests and complete GPU verifier pass again (maximum oracle error
0.12763977, unchanged). The engine and worker now total 446 lines.

Three warmups and 20 samples per variant, alternating baseline/candidate order.
Mixed-length candidates only; both variants share one model. These are scorer
measurements, separate from the baseline public-worker measurements above.
All raw records, the exact executed script, source hashes and reproduction steps
are in [the follow-up bundle](results/l40s-token-copy-20260929/README.md).

| Candidates | State | Batch | Baseline p50 / p95 ms | Reuse p50 / p95 ms | p50 change |
| ---: | --- | --- | ---: | ---: | ---: |
| 16 | short | 32 | 23.55 / 33.97 | 22.65 / 23.67 | -3.81% |
| 16 | short | all | 23.22 / 25.43 | 22.30 / 23.51 | -3.94% |
| 16 | long | 32 | 36.19 / 36.88 | 35.73 / 36.20 | -1.28% |
| 16 | long | all | 36.92 / 37.39 | 36.16 / 37.09 | -2.07% |
| 255 | short | 32 | 231.45 / 233.25 | 224.89 / 227.23 | -2.83% |
| 255 | short | all | 213.76 / 218.63 | 208.38 / 210.25 | -2.52% |
| 255 | long | 32 | 360.16 / 371.50 | 350.77 / 366.43 | -2.61% |
| 255 | long | all | 329.65 / 334.40 | 323.97 / 327.89 | -1.72% |

Peak allocation is identical in six cells; in the two 255-candidate/batch-32
cells it increases by 4,608 bytes (below 0.00021%). Small percentile differences
are descriptive: the first row's p95 reduction is not a general tail-latency claim.
The previous full matrix and frontend results remain measurements of f46f458;
they were not replaced or relabelled as runs of the optimized source.
