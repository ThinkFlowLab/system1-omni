"""Native GPU server lifecycle and HTTP checks. Only manages its own child process."""

import concurrent.futures, json, os, signal, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

ckpt, bundle, out = sys.argv[1:]
out = Path(out)
log = out.with_suffix(".log").open("w")
proc = subprocess.Popen(
    ["target/release/omni-laya", ckpt, bundle, "127.0.0.1:18088"],
    stdout=log,
    stderr=log,
)
base = "http://127.0.0.1:18088"
cases = json.loads(Path(__file__).with_name("fixtures.json").read_text())
req = next(c["request"] for c in cases if c["name"] == "short_1")


def call(path, body=None):
    start = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                base + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=40,
        ) as r:
            return r.status, r.read(), (time.perf_counter_ns() - start) / 1e6
    except urllib.error.HTTPError as e:
        return e.code, e.read(), (time.perf_counter_ns() - start) / 1e6


try:
    deadline = time.monotonic() + 120
    while True:
        assert proc.poll() is None, "server startup failed"
        try:
            if call("/health")[0] == 200:
                break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("readiness")
        time.sleep(0.2)
    status, response, _ = call("/v1/systemone", req)
    assert status == 200
    for _ in range(10):
        assert call("/v1/systemone", req)[0] == 200
    c1 = [call("/v1/systemone", req) for _ in range(50)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        c8 = list(ex.map(lambda _: call("/v1/systemone", req), range(80)))
    assert all(
        s == 200 and json.loads(b) == json.loads(response) for s, b, _ in c1 + c8
    )
    bad = {
        "model": "english",
        "state": "",
        "questions": {
            str(i): {
                "type": "choice",
                "instructions": "Pick",
                "criteria": [str(j) for j in range(129)],
            }
            for i in range(16)
        },
    }
    assert call("/v1/systemone", bad)[0] == 400
    assert call("/v1/systemone", req)[0] == 200 and call("/health")[0] == 200
    maps = Path(f"/proc/{proc.pid}/maps").read_text()
    out.with_suffix(".maps").write_text(maps)
    assert "libpython" not in maps and "libtorch" not in maps
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=15) == 0
    result = {
        "ready": True,
        "C1_ms": [t for _, _, t in c1],
        "C8_ms": [t for _, _, t in c8],
        "responses_equal": True,
        "oversize_400_worker_survives": True,
        "sigterm_exit": 0,
        "python_torch_absent": True,
    }
    out.write_text(json.dumps(result, indent=2))
    print("HTTP_ACCEPTANCE_PASS", flush=True)
finally:
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=15)
    log.close()
