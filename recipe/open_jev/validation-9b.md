# Open-Jev-9B validation

## Reference comparison, 2026-10-05

The native worker serving the Open-Jev-9B export was compared with the
[Open-Jev reference](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16)
at `3308a15` (uncached) on 253 requests, both with 9B's trained head and saved
temperature (1.8969118766347646). Every request had the same input token count
on both sides.

| Requests | Questions | Same decision | Largest probability difference | Questions above 0.01 |
| --- | ---: | ---: | ---: | ---: |
| M1 to M3: generated `choice` and `score` requests and Open-Jev's examples (22) | 87 | 87 | 0.041 | 5 |
| M4: JevBench single-candidate `noul` tasks (74) | 74 | 74 | 0.068 | 4 |
| M5: the other public JevBench tasks, 3 to 6 candidates (157) | 157 | 155 | 0.104 | 25 |

Both changed decisions are close calls on both sides. One is a four-way choice
whose top two options are at 0.403/0.390 native and 0.397/0.400 in the
reference; the other is a three-way choice split almost evenly between two
options, 0.52/0.48 native and 0.44/0.56 in the reference. The largest difference
is on a four-way choice of about 2,240 tokens per candidate. Differences grow
with prompt length: across JevBench, the mean of each question's largest
difference is 0.012 above 2,000 tokens per candidate and 0.003 below 500.

The final-norm hidden state at each candidate's last token, the input to the
scalar head, was also compared for all 2,157 candidates:

| Requests | Candidates | Relative L2: median | p99 | largest | Head output: median | p99 | largest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 to M3 | 1,382 | 0.0093 | 0.020 | 0.027 | 0.024 | 0.18 | 0.36 |
| M4 | 74 | 0.014 | 0.036 | 0.050 | 0.051 | 0.43 | 0.87 |
| M5 | 701 | 0.011 | 0.028 | 0.063 | 0.038 | 0.39 | 1.49 |
| All | 2,157 | 0.0096 | 0.023 | 0.063 | 0.028 | 0.21 | 1.49 |

The relative L2 difference is ‖native − reference‖ / ‖reference‖ per candidate.
The head output is the scalar before temperature, computed from each side's
hidden state with the same FP64 head. These differences come from two different
implementations: the native worker runs the LoRA adapter merged into BF16
weights on its CUDA kernels, while the reference applies the adapter unmerged
through PEFT, uses the PyTorch Gated DeltaNet path and pads candidates into
batches of 8. The median relative L2 difference is 0.009 to 0.010 in each of
three prompt-length ranges (below 500 tokens per candidate, 500 to 2,000, and
above 2,000), while the tail of the head output difference grows with length
(p99 0.18 below 500 tokens, 0.38 above 2,000), as do the probability
differences above.

These results are for 9B on one GPU. They make no claim about 27B.

### Workload

- **M1:** one `choice` question with 2, 8, 32, 128 or 255 candidates, over
  states of about 256, 1,024 or 3,072 tokens built from the text of Open-Jev's
  example states (15 requests).
- **M2:** the four `examples/workflows` requests and `examples/community/support_28`
  at the reference revision, 7 to 50 candidates each (5 requests).
- **M3:** one `score` question with 5 or 10 levels over a 1,024-token state
  (2 requests).
- **M4 and M5:** the 231 public tasks of
  [JevBench](https://github.com/fstandhartinger/jevbench) at
  `f8ce71361165846101d02ebc83ad44e47ae44fc3`, one question each: the 74 `noul`
  tasks used in the [H200 validation](validation.md), and the 157 tasks with 3 to
  6 candidates. JevBench's license allows publishing aggregate results only, so
  requests and responses are not included.

### Controls and reproduction

- One NVIDIA RTX 6000 Ada (compute capability 8.9, 48 GB, 300 W limit), CUDA 13.2,
  BF16, one request at a time. The card ran near 1 GHz at its power limit during
  the long requests; timing is not part of this comparison.
- Native: this change's Rust sources and `Cargo.lock` on `main` `7f39ac4`
  (unchanged since the measurement), with the CUDA library built by
  `src/backends/cuda/qwen3_5/build.sh <dir> 89`, eager execution
  (`CUA_S1_GRAPH` unset), requests over HTTP to the worker. The worker used
  15,874 MiB of device memory after warmup and 16,672 MiB after all requests.
- Export: [`export_merged.py`](export_merged.py) on the pinned base and checkpoint
  in the [native recipe](native.md), with the default 4096-token limit. It
  produced the same bytes as the export used for these measurements, in all 10
  files.
- Reference: `jev.serving.load_predictor` on the 9B checkpoint package, batch
  size 8, prefix cache off, with Torch 2.14.0+cu130, Transformers 5.10.2,
  PEFT 0.19.1 and Accelerate 1.13.0. Flash Linear Attention is not installed,
  so Gated DeltaNet runs on the PyTorch path, as in the H200 HF baseline.
- Hidden states: on the native side, `Model::forward` with the worker's own
  prompt rendering and tokenization; on the reference side, the input to its
  scalar head, recorded in a separate run with the same predictor settings.
  Through each side's head, the native hidden states reproduce the worker's
  probabilities and the recorded reference ones reproduce the compared reference
  probabilities, both to within 1e-15.

Request manifests, raw responses, hidden-state dumps and scripts are kept
locally, outside this repository.
