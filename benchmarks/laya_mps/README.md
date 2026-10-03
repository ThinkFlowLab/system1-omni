# Laya on Apple Silicon: benchmark scripts

Scripts behind the numbers in the [Apple Silicon recipe](../../recipe/laya/apple-silicon.md). Each run writes raw
JSONL to `results/` (kept out of the repository); `report.py` builds the tables from it.

| file | purpose |
| --- | --- |
| `workloads.jsonl` | the fixed inputs: W1–W6 are timed, P* are for answer comparison only. Tokens per row with Laya's tokenizer: W1 68, W2 198, W3 484, W4 68/48/47, W5 40–68, W6 47 |
| `bench_inproc.py` | Laya in-process (no HTTP): load, warmup, first request, warm latency, memory |
| `bench_http.py` | a `/v1/systemone` server, optionally started by the script and optionally behind the frontend: time to ready, first request, warm latency, throughput |
| `paired.py` | two configurations compared request by request, both alive at once: two worker flag sets, or two running servers (e.g. a worker directly and through the frontend) |
| `profile_mps.py` | where a request's time goes on MPS |
| `report.py` | tables from the JSONL, including the run-to-run gate and the answer comparison against a reference config |
| `env.py` | shared: versions, checkpoint, hardware, power and load recorded with each run; memory footprint |

## Run

From the repository root, in the recipe's environment (`.venv`), with the frontend built:

```sh
python benchmarks/laya_mps/bench_inproc.py --device cpu --config C1 --run m1
python benchmarks/laya_mps/bench_inproc.py --device mps --config C2 --run m1
python benchmarks/laya_mps/bench_http.py --config C3 --run m1 --spawn .venv/bin/laya-serve
python benchmarks/laya_mps/bench_http.py --config C4 --run m1 --url http://127.0.0.1:8080 \
  --frontend target/release/omni-jev --spawn .venv/bin/laya-serve
python benchmarks/laya_mps/bench_http.py --config C3w --run m1 \
  --spawn .venv/bin/python -m frontend.laya_mps --device {device} --model {model} --port {port}
python benchmarks/laya_mps/bench_http.py --config C3o --run m1 \
  --spawn .venv/bin/python -m frontend.laya_mps --device {device} --model {model} --compile --weights fp16 --port {port}
python benchmarks/laya_mps/report.py benchmarks/laya_mps/results/*_m[0-9].jsonl --ref C1
```

Repeat with `--run m2` for a second measured run. Runs refuse to start on battery power or above a
1-minute load average of `--max-load` (default 2) unless labelled `--run feasibility`. Memory is the
process's physical footprint, which on Apple Silicon includes MPS allocations.

Two configurations compared request by request, which holds up under background load better than
separate runs:

```sh
python benchmarks/laya_mps/paired.py --run p1 --a "" --b "--compile --weights fp16"
python benchmarks/laya_mps/paired.py --run f1 --a-url http://127.0.0.1:8000 --b-url http://127.0.0.1:8080
python benchmarks/laya_mps/paired.py --summarize benchmarks/laya_mps/results/paired_p1.jsonl
```

## Results

The measured runs on an M1 Pro are published as assets of one release on the fork,
<https://github.com/cacheline999/system1-omni/releases/tag/laya-mps-results-2026-09-28>:

| asset | contents | sha256 |
| --- | --- | --- |
| `laya-mps-reports-2026-10-01.tar.gz` | the tables: baseline report and parity, frontend overhead, paired fp16, paired all optimizations | `e857da5082da4983e07a91104b20f84fb1ffd7994c56123e0a832e0d7a870cea` |
| `laya-mps-results-2026-09-28.tar.gz` | raw JSONL of the baseline runs (C1–C4, C3w, C3s) | `611ed30707ac8c98875b5aa5382360b5a7d760da166d61c626eb07ebe1ee6404` |
| `laya-mps-paired-fp16-2026-09-30.tar.gz` | raw JSONL of the paired fp16 runs | `cc0d6f5bda6e3e0ee1f40c6966f84429e902a2f585b8e1ee33658a9be139326e` |
| `laya-mps-paired-all-2026-09-30.tar.gz` | raw JSONL of the paired all-optimizations runs | `25d1b7bc9d6dff7173f9b972ebd0204f8fb4e2089fe2d27924d48ba6e589780c` |

Extract the raw JSONL into `results/` and run `report.py` or `paired.py --summarize` on it to rebuild
the tables. `C3s` in the baseline runs is an earlier compile mode that compiled one-question requests
only; `--compile` does the same for them and adds the encoder for several questions.
