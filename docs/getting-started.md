# Your first decision on CPU

[中文入口](getting-started.zh.md) · [Recorded demo](../recipe/laya/validation.md#decision-demo) ·
[Supported models and hardware](supported-models.md)

An agent can use a decision model to route a support ticket, score its urgency,
or check whether the customer asks for a refund. This walkthrough sends a real
refund decision through the Rust frontend to the **upstream LAYA English worker
on CPU**. It is the recommended first path because it needs no GPU, weight export
or CUDA build and already supports `choice`, `score` and `noul` text questions.
It does not exercise native Rust model execution or image, audio or video inference.

## Before running commands

| Requirement | This walkthrough |
| --- | --- |
| Host | Linux x86_64 CPU; four PyTorch threads, no accelerator required. On an Apple Silicon Mac, change the two install commands as in [Troubleshooting](#troubleshooting-and-next-steps); for the Apple GPU, use the [MPS recipe](../recipe/laya/apple-silicon.md). |
| RAM and disk | Budget 8 GB host RAM and 6 GB free disk for the environment, English checkpoint and Rust build, excluding installation of the toolchains themselves. These are planning allowances for short text, not measured minimums; the [reproduction host](../recipe/laya/validation.md) had more memory. Longer inputs and extra checkpoints need more. |
| Tools | Git, curl, Python **3.12** with `venv`/pip, Rust stable with Cargo, and a C compiler/linker. On Debian/Ubuntu the compiler tools are in `build-essential`. The reproduction used Python 3.12.13 and Rust/Cargo 1.98.1. |
| Dependencies | `laya[serve]==0.3.20`, CPU-only `torch==2.8.0+cpu`, `transformers==4.55.0`; other tested direct dependencies are pinned in [requirements-cpu.txt](../recipe/laya/requirements-cpu.txt). This is separate from the MPS environment. |
| Model download | Public [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya); about **846 MB** for the English weights, tokenizer and encoder configuration. The worker downloads only the English checkpoint when preloaded as below. No Hugging Face token is required for this public model. |
| Network and ports | Access to GitHub, PyPI, the PyTorch CPU wheel index and Hugging Face for initial preparation. Free localhost ports **8000** (worker) and **8080** (frontend); use three terminals. |

The [recorded runs](../recipe/laya/validation.md) used model revisions
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (cached) and
`7b928d828b7b0e022f929d9bd2e44165aa270148` (fresh Hub cache). `laya-serve` 0.3.20 downloads the
Hub's default revision and has no revision flag; a later download may produce
different probabilities. Record the resolved revision in step 5. Installation,
downloads, compilation and loading are preparation, not inference latency.
There is no promised setup duration or CPU latency benchmark here.

## 1. Install and build — terminal 1

Start from a directory where you want the checkout, then keep all three terminals
in this repository root. If you already have a checkout, start at the `venv` command.
If `.venv` already exists, verify its packages before reusing it.

```sh
git clone https://github.com/ThinkFlowLab/system1-omni.git
cd system1-omni
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r recipe/laya/requirements-cpu.txt
cargo build -p omni-jev --release --locked
```

Building only `omni-jev` prepares the HTTP frontend; it needs no CUDA toolkit.

## 2. Start the worker — terminal 1

```sh
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu \
LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 \
  .venv/bin/laya-serve
```

Keep this process running. On the first start it downloads the model to the
Hugging Face cache, then loads it before listening. Later starts reuse the cache.
Wait for `Uvicorn running on http://127.0.0.1:8000`; download or load failures
must be resolved in this terminal before proceeding.

## 3. Check the worker and start the frontend — terminal 2

From the same repository root:

```sh
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8000/health
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

The health response must be `{"status":"ok","loaded":["english"],"device":"cpu"}`.
The frontend prints `omni-jev listening on 127.0.0.1:8080` and
`forwarding to http://127.0.0.1:8000/`. Keep both services running.
Health confirms the model is loaded; plain `laya-serve` does not warm up a
forward pass before health, so the first decision may still initialize inference.

## 4. Make a decision — terminal 3

```sh
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8080/health
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

The frontend health response matches the worker's. The decision returns HTTP
200 with `answers.refund.type` equal to `noul` and a number in
`answers.refund.noul`. In the [recorded run](../recipe/laya/validation.md#decision-demo)
the answer subtree was:

```json
{"refund":{"type":"noul","noul":0.8364,"confidence":0.8364,"answer_confidence":0.8364,"action":{"act_probability":1.0}}}
```

`noul` is the model's probability-like yes value for the question. The full
response also includes `model`, `usage` and `routing`; `routing.model` is
`english`. Exact probabilities are an observed example, not an accuracy or
calibration guarantee. This English checkpoint demo uses English inputs even
when following the Chinese entry page.

## 5. Check all question types and record your setup — terminal 3

```sh
.venv/bin/python recipe/compare_with_backend.py --model english \
  --backend http://127.0.0.1:8000 --frontend http://127.0.0.1:8080
git rev-parse HEAD
.venv/bin/python -m pip freeze
.venv/bin/python - <<'PY'
from huggingface_hub import scan_cache_dir
for repo in scan_cache_dir().repos:
    if repo.repo_id == "convaiinnovations/laya":
        print("cached model revisions:", sorted(r.commit_hash for r in repo.revisions))
        for rev in repo.revisions:
            if "main" in rev.refs:
                print("cached main:", rev.commit_hash)
PY
```

Expect `PASS health`, `PASS department`, `PASS urgency`, `PASS refund` and
`PASS combined`, each with `status 200 -> 200`. The script also prints the
returned JSON. It checks direct versus frontend status, content type and body,
with equal parsed `answers` accepted when serialization or usage differs. This
checks transport consistency and all three question types, not task accuracy.
The cached `main` revision identifies the default checkpoint for this run;
inspect it before another process updates that cache reference.

Stop each server with Ctrl-C in its own terminal when finished. For an independent
first-use report, include OS/CPU/RAM, repository SHA, package versions, model
revision, the commands and changes you made, and the first failing or confusing
step in [issue #86](https://github.com/ThinkFlowLab/system1-omni/issues/86).
Separate any download/build/startup durations from first-request and warm latency.

## Troubleshooting and next steps

- **`cargo` or `python3.12` missing:** install the prerequisite toolchain before
  step 1. If Python lacks `venv`/pip, install those for that interpreter, or use
  `uv venv --python 3.12 --seed .venv` in place of `python3.12 -m venv .venv`.
- **Apple Silicon Mac:** PyTorch has no `torch==2.8.0+cpu` wheel for macOS, so both
  install commands in step 1 fail with `No matching distribution found for torch==2.8.0+cpu`.
  The compiler and linker come from the Xcode Command Line Tools (`xcode-select --install`).
  Install the plain macOS wheel and the other pins instead; on an M5 Pro with macOS 26.6
  the other steps then ran unchanged and printed the same answers as the
  [recorded run](../recipe/laya/validation.md).

    ```sh
    .venv/bin/python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cpu
    sed 's/+cpu$//' recipe/laya/requirements-cpu.txt | .venv/bin/python -m pip install -r /dev/stdin
    ```

- **Connection refused / frontend 502:** check the worker's terminal and its
  direct health URL first. The frontend can listen even while its worker is absent.
  A 504 means the worker exceeded the frontend timeout; see the
  [frontend configuration](../src/frontend/README.md).
- **Port already in use:** select a free worker/frontend port and update the
  matching bind variables, backend URL and every curl/comparison URL. Do not
  stop another service to free its port.
- **Download failure:** check Hugging Face connectivity and free cache storage;
  retry the worker. A `Fetching 5 files` progress bar that stops advancing for
  minutes is a failure too: stop the worker with Ctrl-C and start it again.
  `HF_HUB_OFFLINE=1` is suitable only after the complete checkpoint is cached;
  it cannot prepare an empty cache.
- **Temperature warning:** this checkpoint/runtime combination warns that a
  `choice:11+` calibration temperature is clamped. That warning did not prevent
  this demo or the two-option comparison; confidence for affected entries should
  be treated as uncalibrated.

For Apple GPU execution, use the [LAYA MPS recipe](../recipe/laya/apple-silicon.md).
For accelerated native serving, use [Open-Jev's CUDA recipe](../recipe/open_jev/native.md)
after checking its much larger hardware and export requirements. Its
[H200 performance result](../recipe/open_jev/validation.md) is a separate
single-candidate workload, not a measurement of this CPU walkthrough.
The [support matrix](supported-models.md) distinguishes implemented paths,
hardware validation and planned work.
