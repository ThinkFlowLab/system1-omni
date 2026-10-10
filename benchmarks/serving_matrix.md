# Frozen HTTP serving matrix

`serving_matrix.py` is a Python 3.11+ standard-library client for an already
running worker or frontend. It measures closed-loop HTTP concurrency and checks
every response against a frozen exact oracle. It does not launch models or reserve
a GPU. The [example plan](serving_matrix.example.json) uses synthetic CPU-server
responses and explicitly records N/A for GPU fields. Replace its requests,
oracles, endpoints and identity with your deployment before measuring.

```sh
python3 benchmarks/serving_matrix.py /path/to/frozen-plan.json \
  --output /tmp/matrix-new-run
```

The output directory must not exist. Validation precedes output creation and
network requests. Exit 0 means all phases passed; exit 1 means an HTTP,
correctness, timeout or transport failure was recorded.

## Plan schema

- `endpoint`: HTTP(S) inference URL, without URL credentials or fragments.
- `health_endpoint`: optional GET URL; defaults to `/health` on the inference
  origin. Health requires HTTP 200; its body is retained for inspection.
- `cases`: nonempty list with unique nonempty `name` values. Each case contains a
  JSON-object `request` and exactly one of `expected_response` (type-strict JSON)
  or `expected_response_text` (exact UTF-8 bytes, including whitespace).
- Alternatively a case has a nonempty `variants` list instead of request/oracle
  fields. Every variant has a unique `name`, request and oracle. Waves cycle
  variants by dispatch index, preserving planned order across configurations.
- `requests_per_case`: positive integer, applied per measured round.
- `concurrency`: unique positive integers no greater than `requests_per_case`.
- `repetitions` and `warmup_per_case`: integers >= 2, both default to 2 when
  omitted. The frozen plan predeclares the budget; do not extend it after seeing
  results. Effective defaults are recorded in `config.json`.
- `timeout_seconds`: finite positive deadline for a complete exchange.
- `metadata`: requires nonempty `gpu`, `gpu_ids`, `driver`, `cuda`, `precision`,
  `model_revision`, `runtime_revision`, `cache_policy`, `cuda_evidence` and
  `reservation` and `host` (server node). Record pinned revisions, the actual
  reserved device and node, not just a shared service URL. CPU
  fixtures must explicitly record N/A with a reason. The client checks presence;
  it does not independently verify these declarations or prove CUDA execution.

JSON numbers must be finite. Type-strict matching distinguishes `true` from `1`
and integers from floats, preserves array order and ignores object-key order.
Use exact-byte matching when formatting matters. The runner never alters model
aliases, injects token counts, truncates prompts or invents oracles. Capture a
correct response before freezing the plan.

## Responses with dynamic timing fields

The native Open-Jev response includes per-request `metadata.inference_seconds`;
JEMM includes `usage.latency_ms`. A captured whole-body oracle cannot match these
values on later requests. Declare `ignore_paths` at the plan's top level to
exclude only those volatile leaf values from JSON comparison:

```json
{"ignore_paths": ["/metadata/inference_seconds"]}
```

For JEMM use `/usage/latency_ms` instead. Paths are RFC 6901 JSON Pointers; `/`
inside a key is encoded `~1`, `~` is `~0`, and arrays use canonical zero-based
indices. The list must be unique. Every path must resolve in every variant's
oracle to an existing primitive leaf. Root, object/array exclusions, malformed
escapes and invalid indices are rejected before output/network access.

Excluded fields must still exist in the actual response with the same JSON type.
All other fields remain type-strict and exact, including model identity, answers,
token usage, key sets and array order. Declare only volatile diagnostics; do not
exclude decisions or token counts to conceal disagreement. No exclusions are
inferred automatically. Exclusions are incompatible with exact-byte oracles.
The list is bound by `plan_sha256`; raw response bytes and the oracle snapshot
remain unmodified, so the original values can still be audited.

## Authentication

Set `OMNI_JEV_TEST_TOKEN` in the environment to send `Authorization: Bearer …`
on health and every inference phase, matching `bench.py`. It must be printable
ASCII and is read once before dispatch. Keep credentials out of the plan and
metadata; the client does not save request headers or the token. An empty/unset
variable sends no Authorization header. Redirects, environment proxies, pooling
and retries are disabled. Use HTTPS for authenticated remote endpoints.

## Phases and metrics

One health request precedes one readiness request per variant, the predeclared
excluded warmups per variant, then one excluded feasibility wave at each concurrency
and the predeclared measured rounds (default two). Each worker sends its next request after the previous
exchange completes. There is no arrival-rate or queue-delay simulation. A failure
stops new dispatch while already admitted work drains.

Client wall time includes DNS/TCP/TLS connection establishment, request serialization,
complete response read, oracle validation and connection close. Each request opens
a fresh connection; `connect_seconds` reports the connect call separately (null
if it fails). TLS handshakes are included. This policy differs from `bench.py`'s
keep-alive pool, so their latency/throughput figures are not directly comparable.
Evidence encoding and sink writes affect client dispatch and complete-wave elapsed
time, but are outside the recorded exchange latency.

Round requests/s and decisions/s use actual successful counts
and whole-round elapsed time; planned, attempted and successful denominators are
separate. Decisions count elements of `answers`. Successful p50/p95 use nearest
rank; **p95 is the maximum when fewer than 20 samples are present**, including the
four-sample example. Failure latencies and kinds are retained separately. First-use, capture and
loading costs must be evaluated separately from warmed results.

Whole-wave rates are explicitly labelled `complete_wave_including_ramp_and_drain`.
For a complete measured round with n > 2c, `steady_state` additionally reports an
interior completion window: sort retained monotonic completion timestamps, exclude
the first and last c completions, and measure from the c-th completion boundary
to the last retained completion. Count only the n−2c interior requests/decisions.
Short (n <= 2c), failed or zero-duration windows have `available: false` and a reason.
In the four-request example, concurrency 2 has no interior-window estimate;
concurrency 1 can report a two-request window, which is too short to establish stationarity.

Prefer at least 16 requests per worker for measurements (for example 2048 requests
at concurrency 64); short plans remain useful smoke checks. The interior estimator
removes boundary completions, not every source of ramp effects or variation. It
does not prove stationary traffic or isolate server-only throughput.

`round_aggregates` groups each case/concurrency and reports min/median/max and
relative spread `(max-min)/median` for complete-round rates, p50/p95 and available
interior rates. Failed rounds are counted explicitly and excluded from aggregates;
absent metrics are null. When all values are zero, relative spread is zero.
These are descriptive summaries, not confidence intervals. Stable back-to-back
rounds do not establish stability across invocations, hosts or days.

Output includes the byte-exact `plan.json`, plan/runner SHA-256 in `config.json`,
UTC `started_at`/`finished_at`, client hostname/platform/Python version/CPU count,
connection policy and effective budget in `config.json`, health response,
`responses.jsonl` with phase/variant, monotonic completion time and raw base64/text/status/
error data, and `summary.json` with completeness and each round's metrics.
Completion means the frozen workload passed. It does not establish model quality
or a universal speedup. Preserve failures and repeated rounds; do not silently
relax an oracle or extend the run budget after failure.

## Reproduce the example without a model

Run from the repository root with Python 3.11+:

```sh
python3 - <<'PY'
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from benchmarks.serving_matrix import run
from tests.benchmarks.test_serving_matrix import server
with server() as (origin, _), TemporaryDirectory() as scratch:
    plan = json.loads(Path("benchmarks/serving_matrix.example.json").read_text())
    plan.update(endpoint=origin + "/v1/systemone", health_endpoint=origin + "/health")
    path = Path(scratch) / "plan.json"
    path.write_text(json.dumps(plan))
    assert run(path, Path(scratch) / "output")["complete"]
PY
python3 -m unittest discover -s tests/benchmarks -p 'test_*.py' -v
```

Tests cover exact oracles, mixed variants, bounded concurrency, stop/drain,
deadlines, provenance validation and authenticated health/inference. They make
no GPU or production-throughput claim.
