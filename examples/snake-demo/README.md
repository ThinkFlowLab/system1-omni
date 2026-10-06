# Snake demo: Laya played through a system1-omni backend

![12 decisions/s, recorded against the native worker on one H800](https://github.com/linear3735/system1-omni/releases/download/demo-assets/snake-demo.gif)

![same seed at max speed: 2400 steps in 7.7 s](https://github.com/linear3735/system1-omni/releases/download/demo-assets/snake-topspeed.gif)

The snake client from [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)
(Apache-2.0, see `LICENSE-laya-mlx` and `NOTICE`), driven by a `/v1/systemone`
backend instead of a local MLX runtime. `snake/policy.py::decide` is verbatim
upstream; only the backend behind `predict(state, questions)` is replaced
(`shim.py`). Each step sends one request with three questions: a `choice` over
the four directions and two `noul` probes for dead-end risk and food
reachability. The cycle-safety shield is unchanged.

## Run

Needs Python 3.10+ with `rich`, and a backend serving `/v1/systemone`: the
native Hopper worker ([recipe/laya/native](../../recipe/laya/native/README.md)),
`laya-serve` on CPU/CUDA ([recipe/laya](../../recipe/laya/README.md)), or the
MPS worker on Apple Silicon
([recipe/laya/apple-silicon.md](../../recipe/laya/apple-silicon.md)).

```sh
PYTHONPATH=. python -m snake --backend http://127.0.0.1:8000 --model english
```

Key flags: `--fps N` (paced decision rate), `--max-speed`, `--headless
--steps N --record run.jsonl`, `--unassisted`.
Export a recording to MP4/GIF:

```sh
PYTHONPATH=. python -m snake export run.jsonl --output run.mp4 --seconds 40 --gif run.gif
```

Recordings play back at their real recorded speed (`playback_speed: 1`); the
renderer writes a sidecar JSON with the recording hash. `tools/make_endcard.py`
renders the score-board end card from session summaries.

## Reference run: native worker on one H800

Recorded against the native worker (`omni-laya`, SM90a, BF16) behind
`omni-jev`, client co-located over localhost HTTP, checkpoint
`convaiinnovations/laya` @ `55cf4c4`. Three 2400-step sessions, zero deaths:

| Session | duration | decisions/s | score | shield interventions |
|---|---|---|---|---|
| seed 7 | 7.67 s | 312.8 | 55 | 9 |
| seed 23 | 7.67 s | 313.1 | 59 | 11 |
| seed 91 | 7.64 s | 314.0 | 59 | 3 |

Mean per-decision latency 2.78 ms including frontend and local HTTP; not
comparable to the CLI matched-batch p50 in the native recipe. `fps`-paced
recordings exist for viewing only (12 fps followable, 60 fps fluid). The same
seed reproduces the same trajectory (deterministic forward).

Verified with `recipe/compare_with_backend.py` on the exact stack used for the
recordings. Session summaries and sidecars are not committed here; they live
with the recordings in the producing run.
