# LAYA CPU reproduction and demo

Agent-run smoke reproduction on **2026-10-05** for
[issue #86](https://github.com/ThinkFlowLab/system1-omni/issues/86).
This report verifies the documented worker/frontend path on a prepared host,
followed by a fresh virtual-environment install probe on that same host. **An independent contributor's first-use reproduction is still pending.**
It does not establish clean-machine install time, CPU latency, or model accuracy.

## Frozen setup

- Repository code: `f594d7dfc4c2bef812e23f7ed73573be9625b287` (upstream `main`).
  This contribution changes documentation, navigation, the CPU dependency list
  and demo artifacts only; serving/model code was unmodified during the run.
- Host: `dedicated-developjob-8gpu2-a029z-64896bc8cf-8p2lw`, Ubuntu 22.04.5,
  Linux x86_64, Intel Xeon Platinum 8480C, 224 logical CPUs, about 2 TiB RAM.
  Execution used the personal `hsliu2` account and CPU-only PyTorch;
  **no GPU device work** was performed. The worker used `LAYA_THREADS=4`.
- Python 3.12.13, rustc/Cargo 1.98.1. Existing CPU environment:
  `laya==0.3.20`, `torch==2.8.0+cpu`, `transformers==4.55.0`,
  `tokenizers==0.21.4`, `safetensors==0.8.0`, `huggingface-hub==0.36.2`,
  `numpy==2.5.3`, `fastapi==0.141.1`, `uvicorn==0.54.0`.
  These direct versions are in [requirements-cpu.txt](requirements-cpu.txt).
- English checkpoint: [convaiinnovations/laya at
  55cf4c4](https://huggingface.co/convaiinnovations/laya/tree/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851).
  Cached weights, tokenizer and encoder config totaled **846,195,574 bytes**.
  `HF_HUB_OFFLINE=1` reused this complete snapshot; no model files were downloaded.
  `laya-serve` has no revision flag, so the normal online walkthrough follows
  the Hub default revision and asks the tester to record it.

## Commands and results

The [walkthrough](../../docs/getting-started.md) describes a fresh environment.
The initial smoke run reused `/home/hsliu2/tmp/venvs/looped-serving-cpu`, linked
as `.venv` in the task checkout. `uv pip install --dry-run --python
.venv/bin/python -r recipe/laya/requirements-cpu.txt` required no changes.
That reused environment lacks pip, so its package capture used `uv pip freeze`.

A subsequent install probe removed only the task-owned symlink and ran the
walkthrough's `python3.12 -m venv .venv`, CPU PyTorch pip install and
`pip install -r recipe/laya/requirements-cpu.txt` in a new environment. All
succeeded; `.venv/bin/python -m pip check` reported no broken requirements, and
`.venv/bin/python -m pip freeze` captured the installed packages. Pip reused
available wheel caches; this is a fresh environment check, not a clean-machine
installation-time measurement.

```sh
cargo build -p omni-jev --release --locked
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu \
LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 HF_HUB_OFFLINE=1 \
  .venv/bin/laya-serve
# Separate terminal; 8080 was already occupied by another service.
OMNI_JEV_BIND=127.0.0.1:8086 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

The frontend build succeeded (`Finished release profile`, Cargo reported 25.14 s).
That is preparation on this host, not an install-time promise or inference
measurement. Worker health and frontend health both returned HTTP 200:

```json
{"status":"ok","loaded":["english"],"device":"cpu"}
```

The worker emitted a calibration warning for `choice:11+`: the checkpoint's
0.10058280825614929 temperature was clamped to 0.5. It continued to serve.
The demo uses `noul`; the comparison's `choice` has two options. No calibration
or task-accuracy claim is made for these outputs.

## Decision demo

**Actual response from the CPU worker through the frontend**, using a synthetic
refund message. This is a text transcript of the run. Port 8086 is the only
frontend port change from the walkthrough; it avoided the pre-existing service.

```sh
curl --noproxy '*' --fail --silent --show-error http://127.0.0.1:8086/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

HTTP 200, full response preserved in [first-decision.json](first-decision.json):

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "refund": {
      "type": "noul",
      "noul": 0.8364,
      "confidence": 0.8364,
      "answer_confidence": 0.8364,
      "action": {
        "act_probability": 1.0
      }
    }
  },
  "usage": {
    "input_tokens": 40,
    "output_tokens": 0
  },
  "routing": {
    "model": "english",
    "repo": "convaiinnovations/laya",
    "reason": "explicit model='english'",
    "detection": null,
    "workflow": null
  }
}
```

The question returned `noul: 0.8364`, with `routing.model: english` and
`usage.output_tokens: 0`. This is a real model decision, not a stub output;
the decimal is an observed example, not a guaranteed value for other revisions.

## Direct versus frontend checks

```sh
python3.12 recipe/compare_with_backend.py --model english \
  --backend http://127.0.0.1:8000 --frontend http://127.0.0.1:8086
```

```text
PASS health: status 200 -> 200
PASS department: status 200 -> 200
PASS urgency: status 200 -> 200
PASS refund: status 200 -> 200
PASS combined: status 200 -> 200
```

The [full comparison stdout](first-decision-comparison.txt) preserves every
proxied response. In this run, direct and frontend responses matched byte for
byte for all five checks; no usage/parsed-answer fallback was reported. The
comparison request has a longer state than the single-request demo, so its
refund probability is different (`0.916`). These checks establish transport
consistency for the tested inputs, not broad decision quality or latency.

## Fresh environment and empty Hub cache

After the install probe, the same worker command ran with a new task-local
`HF_HUB_CACHE`, without `HF_HUB_OFFLINE`. This left the existing shared model
cache intact. The initial `fresh-hub` directory did not exist. Hugging Face's
other caches and pip wheel caches were not cleared; this is an empty **Hub model
cache** check, not a claim that every download layer was cold.

```sh
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu \
LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 \
HF_HUB_CACHE=/home/hsliu2/tmp/system1-omni-first-decision-evidence/fresh-hub \
  .venv/bin/laya-serve
```

The worker fetched five files and loaded the current English checkpoint at
[7b928d8](https://huggingface.co/convaiinnovations/laya/tree/7b928d828b7b0e022f929d9bd2e44165aa270148)
(`7b928d828b7b0e022f929d9bd2e44165aa270148`), again 846,195,574 bytes in the Hub
snapshot. The [fresh environment package capture](fresh-environment.txt) includes
transitive dependency versions; it records the run, rather than replacing the
walkthrough's direct dependency requirements.

Worker/frontend health and the first decision after readiness returned 200.
The JSON was identical to the demo above, and all five direct/frontend checks
passed again with byte equality. The complete comparison stdout matched
[first-decision-comparison.txt](first-decision-comparison.txt) byte for byte.
This confirms the documented setup commands and real requests on this host;
it is not a latency, accuracy or general cross-revision parity benchmark.
Both task-owned servers were stopped after validation; the pre-existing service
on port 8080 was left running.

## Remaining first-use evidence

- Fresh environment creation and dependency installation passed on the same
  host, with wheel caches reused. An independent contributor's first-use report
  on another host is still needed.
- The large host does not establish the minimum RAM/disk requirement on a laptop.
  The walkthrough's 8 GB RAM / 6 GB disk are planning allowances for short text.
- No startup-to-readiness, first-request or warm inference timing was measured.
  The separate [Open-Jev H200 result](../open_jev/validation.md) does not apply here.
- A contributor unfamiliar with the setup should follow the walkthrough on
  their own host and report OS/CPU/RAM, repository and model revisions, package
  versions, exact commands, readiness/decision outputs and the first blocker in
  [issue #86](https://github.com/ThinkFlowLab/system1-omni/issues/86).
