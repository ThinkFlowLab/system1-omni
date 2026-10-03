# Laya on Apple Silicon: benchmark scripts

Scripts behind the numbers in the [Apple Silicon recipe](../../recipe/laya/apple-silicon.md). Each run writes raw
JSONL to `results/` (kept out of the repository); `report.py` builds the tables from it.

| file | purpose |
| --- | --- |
| `workloads.jsonl` | the fixed inputs: W1–W6 are timed, P* are for answer comparison only. Tokens per row with Laya's tokenizer: W1 68, W2 198, W3 484, W4 68/48/47, W5 40–68, W6 47 |
| `bench_inproc.py` | Laya in-process (no HTTP): load, warmup, first request, warm latency, memory |
| `bench_http.py` | a `/v1/systemone` server, optionally started by the script and optionally behind the frontend: time to ready, first request, warm latency, throughput |
| `paired.py` | two configurations compared request by request, both alive at once: two worker flag sets, or two running servers (e.g. a worker directly and through the frontend) |
| `lengths.py` | first request of new input lengths and the memory it adds, for one or two worker flag sets |
| `late_load.py` | a checkpoint loaded while serving: its first request directly and through the frontend |
| `fallback.py` | Laya's fallback to the CPU on a GPU out-of-memory error, triggered by a lowered MPS memory limit |
| `release.py` | in-process: what releasing PyTorch's MPS caches gives back after many lengths, and what it costs |
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

## Reproduce every number in the recipe

Each claim in the [recipe](../../recipe/laya/apple-silicon.md) comes from one of these commands. A
rerun on another Mac, or under different load, gives other numbers; the comparison each command makes
(A against B in the same run) is what carries over. Every script refuses a measured run (any `--run`
label but `feasibility`) on battery power or above `--max-load`; paired and lengths runs hold up better
than separate ones under the load that remains, because both sides see it. Stop the recipe's worker and
frontend first: the scripts start their own on ports 8000, 8001 and 8080.

| recipe claim | command |
| --- | --- |
| checkpoint download time | `HF_HOME="$(mktemp -d)" python -c "import time, huggingface_hub as h; t = time.time(); h.snapshot_download('convaiinnovations/laya', allow_patterns=['rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*']); print(f'{time.time() - t:.0f} s')"` |
| first request after ready, time to ready, memory: worker against laya-serve, with and without the options | `bench_http.py` C3, C3w and C3o above, then `report.py` (phases and memory tables) |
| first request after ready over many fresh starts | the loop below the table |
| warm latency and answers, with the options against without | `paired.py --run p1 --a "" --b "--compile --weights fp16"` |
| what each option contributes | `paired.py --run p2 --a "" --b=--compile`, `paired.py --run p3 --a "" --b "--weights fp16"`, and fp16 on top of compile: `paired.py --run p4 --a=--compile --b "--compile --weights fp16"` |
| frontend overhead | start a worker on 8000 and the frontend on 8080 as in the recipe, then `paired.py --run f1 --a-url http://127.0.0.1:8000 --b-url http://127.0.0.1:8080` |
| a request after an idle gap | `paired.py --run i1 --a "" --b "--compile --weights fp16" --gap 2 --only W1 -n 30 --discard 2` (also `--gap 0.5`) |
| first request of a new input length; memory growth with the lengths seen | `lengths.py --run l1 --a "" --b "--compile --weights fp16"` (add `--first-words 1 --step 1 --lengths 1000` for every length up to the window) |
| memory released by `torch.mps.empty_cache()` and the cost afterwards | `release.py --compile --weights fp16` |
| a checkpoint loaded while serving, directly and through the frontend | `late_load.py --flags "--compile --weights fp16" --frontend target/release/omni-jev`, and without `--flags` |
| fallback to the CPU on a GPU out-of-memory error | `fallback.py --limit-gb 3.5`, and `--limit-gb 2.5 --flags "--compile --weights fp16"` |
| where the time goes | `profile_mps.py --run p1` |
| two commits compared | start each worker from its own checkout on its own port, then `paired.py --a-url ... --b-url ...` |

Each fresh start is one `bench_http.py` run; the phases table of `report.py` then lists the first request of
every start (one measured run per start, so it needs an idle machine):

```sh
for i in $(seq 23); do
  python benchmarks/laya_mps/bench_http.py --config C3o --run s$i --only W1 -n 1 --discard 1 \
    --concurrency 1 --spawn .venv/bin/python -m frontend.laya_mps --device {device} --model {model} \
    --compile --weights fp16 --port {port}
done
python benchmarks/laya_mps/report.py benchmarks/laya_mps/results/http_C3o_s*.jsonl --ref C3o
```

All scripts are run as `python benchmarks/laya_mps/<script>` from the repository root; `--summarize`
rebuilds a paired or lengths table from its JSONL. The published paired runs were made with the earlier
environment-variable form of the options (`LAYA_WORKER_COMPILE=on LAYA_WORKER_WEIGHTS=fp16`), which the
flags replace one for one.

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
