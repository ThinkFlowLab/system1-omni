# Laya on Apple Silicon

This recipe serves Laya on the GPU of an Apple Silicon Mac (PyTorch MPS) with the worker in
[`src/frontend/laya_mps.py`](../../src/frontend/laya_mps.py), puts the Rust frontend in front of it and
runs the benchmarks. The model-side code is in [`src/models/laya/`](../../src/models/laya/). The
[Laya text worker](README.md) recipe covers plain laya-serve on the CPU.

Validated on an M1 Pro (16 GB, 16-core GPU), macOS 26.1, Python 3.12, `laya[serve]==0.3.20`,
torch 2.14.0 and the `english` checkpoint (`convaiinnovations/laya` at `55cf4c4`), and by another
contributor on an M5 (10-core GPU, 32 GB, macOS 26.5.2). A reviewer ran the tests on an M4 (10-core GPU,
16 GB, macOS 26, Python 3.13), including the contract tests on the GPU. Other M-series Macs have not been
tested.

Run all commands from the repository root.

## Install

Use Python 3.12. If `python3.12` is not on your `PATH` and you have uv, replace the first command below
with `uv venv --python 3.12 --seed .venv` (`--seed` puts pip in the environment).

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r recipe/laya/requirements-mps.txt
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

The last command must print `True`. The standard macOS arm64 wheel of torch includes MPS.

## Start the worker

```sh
PYTHONPATH=src .venv/bin/python -m frontend.laya_mps --device mps --model english --require-device --port 8000
```

First startup downloads the checkpoint (846 MB; 97 s into an empty cache at 8.7 MB/s when measured). The
worker loads the model, runs a warmup
over short, long and multi-question requests, and only then listens on port 8000, so the first
request it accepts is already warm: on an M1 Pro the first request after ready took 70–81 ms, against
0.7–1.1 s from plain laya-serve. `--require-device` makes it exit instead of silently serving on the CPU
when the model cannot be placed on MPS; without it the worker logs a warning and serves from the CPU.
Of laya-serve's environment variables, `LAYA_API_KEY` (bearer authentication) still applies. Those its
launcher reads do not: device, model, host, port and log level are the flags above, and `LAYA_THREADS` and
`LAYA_AUTO_TASK` are not read. The worker warns at startup if any of them is set.

Laya loads another checkpoint when a request names it (`"model": "multilingual"`) or its routing picks it
(a non-English state). The worker prepares that checkpoint the same way inside that first request; other
requests wait behind it, `/health` names it under `preparing` meanwhile and lists it afterwards. On the
M1 Pro that first request took 5–10 s without the options and 19–22 s with `--compile` (plus the download
the first time, 680 MB for `multilingual`). With a second worker holding memory on the same GPU it once
took 71 s; the frontend gives a backend 60 s, so it answered that request with 504 while the worker
finished preparing, and the same request sent again took 35 ms. To be safe, send one request for each further checkpoint straight to the
worker after startup. Each resident checkpoint needs its own memory (see Troubleshooting).

With `--require-device`, a checkpoint that does not land on the requested device is unloaded again and
the request fails with 500. If Laya evicted another checkpoint to make room for it (it keeps two by
default), the worker loads that one again.

Check what it is running on:

```sh
curl -s http://127.0.0.1:8000/health
```

`device` must be `mps` and `device_mismatch` `false`. The response also names the checkpoint and
revision, the weight dtype (`torch.float32`; Laya upcasts the fp16 checkpoint on MPS), the autocast
dtype Laya uses for requests with at least `mps_amp_min_rows` questions, and the warmup time. The device
and dtypes are read on every call: if a request runs out of GPU memory, Laya moves the model to the CPU
and keeps serving, and `/health` then shows `device: cpu` and `device_mismatch: true`. Triggered on the
M1 Pro by lowering PyTorch's MPS memory limit: the request that ran out of memory still returned 200
after 30–73 s, and later 68-token requests took 140–270 ms from the CPU.

### Faster: compile and fp16 weights

```sh
PYTHONPATH=src .venv/bin/python -m frontend.laya_mps --device mps --model english --require-device \
  --compile --weights fp16 --port 8000
```

`--compile` compiles the model during warmup: one-question requests run the whole model compiled,
requests with several questions run the encoder compiled and Laya's decision head as it is.
`--weights fp16` keeps the checkpoint's fp16 weights instead of Laya's fp32 upcast on MPS.

On the M1 Pro, with both workers running and every request sent to each back to back, the two options
together lowered warm p50 against the worker without them by 37–38% for a 68-token one-question
request (about 57 → 35 ms in those runs), 17–20% at 198–484 tokens, 14% for three questions and 18% for
six. Answers stayed within 0.0031 of the fp32 worker's. A worker running on its own uses about 3 GB
with the options instead of 4.2 GB (2.8 GB against 3.5 GB in those paired runs, where the two workers
shared the machine), measured on the six benchmark inputs; see below for how it grows. The price is startup: the worker became ready after 35–39 s instead of 8–10 s, and
its first request after that took 62–78 ms.

On an M5 the same paired comparison gave median ratios of 0.51–0.53 for one-question requests at
47–68 tokens, 0.30–0.33 at 198–484 tokens, 0.37 for three questions and 0.60 for six, most of it from
the fp16 weights, which on that GPU speed up every input even without compile. There the worker was
ready after 19 s instead of 3 s; its first request took 21–36 ms in 21 of 23 fresh starts and 327 and
409 ms in the other two, not yet explained (132–143 ms from plain laya-serve).

### What the warm numbers leave out

The latencies above are for requests sent back to back. Measured on the M1 Pro:

- **Idle gaps.** A request that follows a pause is slower, with or without the options, because the GPU
  has slowed down in the meantime. For a short one-question request (25 ms back to back with the options,
  40 ms without) it took about 50 ms after 0.2–1 s of idle and 105–115 ms after 2–5 s (60–68 ms and
  114–127 ms without the options). This is also why the first request after ready costs more than a warm
  one. An agent that asks once every few seconds sees these numbers, not the back-to-back ones. Keeping
  the GPU busy between requests would avoid it, at the cost of power; the worker does not do this.
- **New input lengths.** The first request of a length the worker has not seen costs about 15 ms more
  once with the options (6 ms without). It is not a recompile (`recompiled_after_ready` stays `false`).
- **Memory grows with the lengths seen.** With `--compile`, PyTorch keeps host memory for every input
  length the compiled model has run, about 5 MB each (fp16 weights alone add little): the 3 GB above became
  3.3 GB after 100 new lengths and 5.2 GB after all 454 that the benchmark's one-question request can
  take (57–512 tokens), more than the 3.7 GB of a worker without the options. `torch.mps.empty_cache()`
  brings it back to about 2.95 GB, near where the worker starts, however much the lengths had added
  (2.95 GB in three runs, 2.90 GB after running the same lengths again and releasing again), and those
  lengths then pay their first-request cost again.

`/health` reports under `compile` how many graphs existed when the worker became ready and how many
exist now; `recompiled_after_ready: true` means a request shape was not covered by the warmup.
`active` is `false` once no model runs the compiled path any more, i.e. after a fallback to the CPU.

Both options apply on the GPU only. After a fallback to the CPU the worker runs Laya's fp32 model
uncompiled, like a worker started without them.

## Start the frontend

In another terminal:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

## Send a request

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

The frontend forwards the worker's response unchanged; `compare_with_backend.py` from the
[Laya text worker](README.md#compare-responses) recipe checks that against this setup as well.

## Test

The tests need `pytest` and `httpx2` (Starlette's `TestClient`; `httpx` works with a deprecation warning);
`requirements-mps.txt` pins them and ruff:

```sh
.venv/bin/python -m pip install -r recipe/laya/requirements-mps.txt
PYTHONPATH=src .venv/bin/python -m pytest tests/laya                    # unit tests, no model
LAYA_CONTRACT=1 PYTHONPATH=src .venv/bin/python -m pytest tests/laya    # plus contract tests against a CPU worker
.venv/bin/ruff format --check src/frontend/laya_mps.py src/models/laya benchmarks/laya_mps tests/laya
.venv/bin/ruff check --select E4,E7,E9,F src/frontend/laya_mps.py src/models/laya benchmarks/laya_mps tests/laya
```

The contract tests start a real worker and check readiness, the three decision types, error responses, and
that its answers match Laya run directly in fp32 on the CPU. On an Apple Silicon Mac, run them against the
GPU as well, without and with the options:

```sh
LAYA_CONTRACT=1 LAYA_CONTRACT_DEVICE=mps PYTHONPATH=src .venv/bin/python -m pytest tests/laya/test_contract.py
LAYA_CONTRACT=1 LAYA_CONTRACT_DEVICE=mps LAYA_CONTRACT_FLAGS="--compile --weights fp16" \
  PYTHONPATH=src .venv/bin/python -m pytest tests/laya/test_contract.py
```

## Benchmark

Stop the worker and frontend first; the benchmark starts its own. The scripts are listed in
[`benchmarks/laya_mps/`](../../benchmarks/laya_mps/README.md), which also lists the command behind each
number in this recipe. A first pass that checks everything runs:

```sh
.venv/bin/python benchmarks/laya_mps/bench_inproc.py --device mps --config C2 --run feasibility
.venv/bin/python benchmarks/laya_mps/bench_http.py --config C3 --run feasibility --spawn .venv/bin/laya-serve
.venv/bin/python benchmarks/laya_mps/bench_http.py --config C4 --run feasibility \
  --url http://127.0.0.1:8080 --frontend target/release/omni-jev --spawn .venv/bin/laya-serve
.venv/bin/python benchmarks/laya_mps/paired.py --run feasibility --a "" --b "--compile --weights fp16"
.venv/bin/python benchmarks/laya_mps/report.py benchmarks/laya_mps/results/*_feasibility.jsonl --ref C2
.venv/bin/python benchmarks/laya_mps/paired.py --summarize benchmarks/laya_mps/results/paired_feasibility.jsonl
```

Runs labelled anything other than `feasibility` refuse to start on battery power or when the
1-minute load average is above 2, so close other heavy applications and plug the Mac in first.

## Troubleshooting

- `device_mismatch: true` at startup, or with `--require-device` the worker exits with
  `asked for mps, english is on cpu`: MPS is not available to this Python. Check the `torch.backends.mps.is_available()` line above (an x86_64
  Python under Rosetta, for example, has no MPS).
- `device_mismatch: true` on a worker that started on MPS: Laya fell back to the CPU after a GPU
  out-of-memory error. It keeps answering, several times slower; free memory and restart the worker
  to get back on the GPU.
- The worker process uses about 4 GB, or 3 GB with fp16 weights (Activity Monitor's Memory column, which
  counts MPS allocations), with one checkpoint loaded; with `--compile` it grows towards 5 GB as it sees
  more input lengths (see "What the warm numbers leave out"); a second one Laya loads later adds its own. On a 16 GB Mac, close other large applications before benchmarking.
- `Address already in use`: another worker or frontend still holds port 8000 or 8080.
