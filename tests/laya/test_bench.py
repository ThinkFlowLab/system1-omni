"""Checks on the benchmark inputs, the run header and the documented commands in recipe/laya. No model."""

import json

import pytest
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks/laya_mps"
sys.path.insert(0, str(BENCH))

import env as bench_env  # noqa: E402


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_header_records_what_makes_two_runs_comparable(monkeypatch):
    record = bench_env.header("some/repo", extra=1)
    assert (
        record["type"] == "env"
        and record["extra"] == 1
        and record["checkpoint"] == "some/repo"
    )
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    assert record["omni_sha"] == head and isinstance(record["omni_dirty"], bool)
    import laya  # noqa: F401
    from importlib.metadata import version

    assert (record["laya"], record["torch"], record["transformers"]) == tuple(
        version(package) for package in ("laya", "torch", "transformers")
    )
    assert record["argv"] == sys.argv and record["python"] == ".".join(
        map(str, sys.version_info[:3])
    )
    assert record["loadavg_1m"] >= 0
    if (
        sys.platform == "darwin"
    ):  # the machine probes use macOS tools; elsewhere they are None (test below)
        assert record["power"] and record["chip"] and record["mem_gb"] > 0
    assert record["utc"].endswith("+00:00")


def test_the_fixed_inputs_span_question_types_lengths_and_option_counts():
    workloads = rows(BENCH / "workloads.jsonl")
    assert len({w["id"] for w in workloads}) == len(workloads)
    questions = [q for w in workloads for q in w["questions"].values()]
    assert {q["type"] for q in questions} == {"choice", "score", "noul"}
    assert {len(q["criteria"]) for q in questions if q["type"] == "choice"} >= {
        2,
        5,
        10,
    }
    bench = [w for w in workloads if w["kind"] == "bench"]
    words = sorted(len(w["state"].split()) for w in bench)
    assert (
        words[0] < 20 and 100 < max(w for w in words if w < 200) and words[-1] > 300
    )  # short, medium, near the window
    assert {len(w["questions"]) for w in bench} >= {
        1,
        3,
        6,
    }  # below and above laya's autocast threshold of 5 rows
    parity = [w for w in workloads if w["kind"] == "parity"]
    assert {q["type"] for w in parity for q in w["questions"].values()} == {
        "choice",
        "score",
        "noul",
    }


DOCUMENTS = [
    "recipe/laya/apple-silicon.md",
    "benchmarks/laya_mps/README.md",
    "src/models/laya/README.md",
]
BENCH_SCRIPTS = (
    "bench_http",
    "bench_inproc",
    "paired",
    "profile_mps",
    "report",
    "lengths",
)
BENCH_SCRIPTS += ("late_load", "fallback")
SCRIPTS = {
    "frontend.laya_mps": "src/frontend/laya_mps.py",
    **{f"{name}.py": f"benchmarks/laya_mps/{name}.py" for name in BENCH_SCRIPTS},
}


def documented_commands():
    """Commands in sh blocks, and inline ones in tables such as the reproduction index."""
    import re

    for document in DOCUMENTS:
        text = (REPO / document).read_text()
        lines = []
        for block in re.findall(r"```sh\n(.*?)```", text, flags=re.S):
            lines += block.replace("\\\n", " ").splitlines()
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        lines += re.findall(r"`([^`\n]*\.py --[^`\n]*)`", prose)
        for command in lines:
            for name, source in SCRIPTS.items():
                if re.search(rf"(^|[/ ]){re.escape(name)}( |$)", command):
                    yield document, command, name, source


def options_of(source):
    import re

    text = (REPO / source).read_text()
    return set(re.findall(r'add_argument\(\s*"(--?[a-z][a-z-]*)"', text))


def test_documented_commands_use_flags_and_files_that_exist():
    import re

    commands = list(documented_commands())
    assert len(commands) >= 25
    worker = options_of(SCRIPTS["frontend.laya_mps"])
    for document, command, name, source in commands:
        assert (REPO / source).exists(), f"{document}: {source}"
        ours = command.split(name, 1)[1].split("--spawn")[0]
        # --a/--b/--flags carry the worker's flags: check those against the worker
        passed = re.findall(r'--(?:a|b|flags)(?: "([^"]*)"|=(\S+))', ours)
        for value in (quoted or bare for quoted, bare in passed):
            assert set(re.findall(r"--[a-z-]+", value)) <= worker, (
                f"{document}: {command}"
            )
        ours = re.sub(r'(--(?:a|b|flags))(?: "[^"]*"|=\S+)', r"\1", ours)
        used = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", ours))
        unknown = sorted(used - options_of(source))
        assert not unknown, f"{document}: {command}: unknown {unknown}"


def test_the_header_degrades_to_none_where_macos_tools_are_missing(monkeypatch):
    run = bench_env._run
    macos_only = ("pmset", "sysctl", "system_profiler")
    monkeypatch.setattr(
        bench_env, "_run", lambda *cmd: None if cmd[0] in macos_only else run(*cmd)
    )
    record = bench_env.header(
        "some/repo"
    )  # what a Linux machine records: no failure, the probes are None
    assert (record["power"], record["chip"], record["gpu_cores"], record["mem_gb"]) == (
        None,
        None,
        None,
        0,
    )
    assert record["omni_sha"] and record["torch"]


# ---------------------------------------------------------------------------------------------- report.py
import report  # noqa: E402

NOUL = {"q": {"type": "noul", "noul": 0.9}}


def test_parity_lists_benchmark_runs_but_not_runs_that_never_answer():
    records = [
        {"type": "env", "config": "C2", "run": "x"},
        {"type": "phase", "config": "C2", "run": "x"},
        {
            "type": "answers",
            "config": "C2",
            "run": "x",
            "workload": "W1",
            "answers": NOUL,
        },
        # profile_mps.py: an env record and its own measurements, never answers
        {"type": "env", "config": "profile-mps", "run": "x"},
        {"type": "sweep", "config": "profile-mps", "run": "x"},
        # a benchmark run that stopped after its warmup, before any answers
        {"type": "env", "config": "C3", "run": "x"},
        {"type": "phase", "config": "C3", "run": "x"},
        # a benchmark run whose answer probes all failed
        {"type": "env", "config": "C4", "run": "x"},
        {"type": "phase", "config": "C4", "run": "x"},
        {
            "type": "answers_error",
            "config": "C4",
            "run": "x",
            "workload": "W1",
            "status": 500,
            "detail": "x",
        },
    ]
    out = report.parity(records, "C2")
    assert "profile-mps" not in out
    assert "| C3 | x | W1 | | | | missing | | | FAIL |" in out
    assert "| C4 | x | W1 | | | | status 500: x | | | FAIL |" in out
    assert out.endswith("0/2 questions within tolerance.")


# ---------------------------------------------------------------------------------------------- paired.py
import paired  # noqa: E402

CHOICE_ANSWER = {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}}


def summary_of(tmp_path, capsys, a, b):
    records = [{"type": "env", "run": "x", "a": "A", "b": "B", "loadavg_1m": 1.0}]
    records += [{"type": "answers", "side": side, "workload": "W1", "answers": ans, "error": None}
                for side, ans in (("A", a), ("B", b))]  # fmt: skip
    records.append(
        {
            "type": "end",
            "health": {"A": {"device": "mps"}, "B": {"device": "mps"}},
            "footprint_mb": {},
        }
    )
    path = tmp_path / "paired_x.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    paired.summarize([str(path)])
    return capsys.readouterr().out


def test_paired_summary_reports_a_question_missing_on_either_side(tmp_path, capsys):
    both = {"q": CHOICE_ANSWER, "r": {"type": "noul", "noul": 0.9}}
    assert "errors ['W1/r']" in summary_of(tmp_path, capsys, both, {"q": CHOICE_ANSWER})
    assert "errors ['W1/r']" in summary_of(tmp_path, capsys, {"q": CHOICE_ANSWER}, both)


def test_paired_summary_reports_answers_of_different_shape(tmp_path, capsys):
    out = summary_of(
        tmp_path, capsys, {"q": CHOICE_ANSWER}, {"q": {"type": "noul", "noul": 0.9}}
    )
    assert "errors ['W1/q']" in out


# --------------------------------------------------------------------------------------------- lengths.py
import lengths  # noqa: E402


def test_only_lengths_the_worker_has_not_run_are_measured():
    window, overhead, warm = (
        512,
        56,
        {182, 432},
    )  # laya's W1 question: tokens = words + 56
    requests = []

    def request(side, words):
        requests.append((side, words))
        return 30.0, min(words + overhead, window)

    measured = lengths.measure(["A", "B"], request, 1, 1, 1000, warm, lambda r: None)
    tokens = [t for _, t in measured]
    assert len(tokens) == len(set(tokens)) == window - overhead - len(warm)
    assert not set(tokens) & warm and max(tokens) == window  # stops at the window
    assert {side for side, _ in requests} == {"A", "B"}


def test_lengths_summary_reports_extra_time_and_memory_per_side(tmp_path, capsys):
    records = [
        {"type": "env", "run": "x", "a": "plain", "b": "fast", "loadavg_1m": 1.0}
    ]
    for side, extra in (("A", 5.0), ("B", 15.0)):
        records += [
            {"type": "length", "side": side, "words": w, "tokens": w + 27,
             "first_ms": 40.0 + extra, "again_ms": 40.0}
            for w in (37, 41, 45)
        ]  # fmt: skip
    records.append(
        {
            "type": "end",
            "footprint_before_mb": {"A": 3400, "B": 2700},
            "footprint_after_mb": {"A": 3410, "B": 2716},
            "recompiled_after_ready": {"A": None, "B": False},
        }
    )
    path = tmp_path / "lengths_x.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    lengths.summarize([str(path)])
    out = capsys.readouterr().out
    assert "| A | `plain` | 3 | 5.0 (5.0) | 3400 → 3410 | None |" in out
    assert "| B | `fast` | 3 | 15.0 (15.0) | 2700 → 2716 | False |" in out


def test_paired_summary_says_how_requests_were_paced(tmp_path, capsys):
    both = {"q": CHOICE_ANSWER}
    assert "back to back" in summary_of(tmp_path, capsys, both, both)
    path = tmp_path / "paired_x.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["gap_s"] = 2.0
    path.write_text("\n".join(json.dumps(r) for r in records))
    paired.summarize([str(path)])
    assert "2.0 s idle before each request" in capsys.readouterr().out


def test_every_option_has_a_command_that_measures_it_alone():
    index = (BENCH / "README.md").read_text()
    assert 'paired.py --run p2 --a "" --b=--compile' in index
    assert 'paired.py --run p3 --a "" --b "--weights fp16"' in index


class FakeLateServer:
    """Stands in for late_load's worker, frontend and HTTP client. The late request answers `status`
    (or raises `error`), after which the late checkpoint is resident or not."""

    def __init__(self, status, resident, error=None):
        self.status, self.resident, self.error, self.posts = status, resident, error, []

    def client(self, url, timeout=120):
        server = self

        class FakeClient:
            def request(self, method, path, body=None, retry=False):
                if method == "POST":
                    server.posts.append((url, timeout))
                    if server.error and len(server.posts) == 1:
                        raise server.error
                    return 20.0, server.status if len(server.posts) == 1 else 200, b"{}"
                models = {"english": {}}
                if server.posts and server.resident:
                    models["multilingual"] = {}
                return (
                    1.0,
                    200,
                    json.dumps({"preparing": [], "models": models}).encode(),
                )

        return FakeClient()


def late_load_with(monkeypatch, status, *argv, resident=True, error=None):
    import late_load

    server = FakeLateServer(status, resident, error)
    monkeypatch.setattr(late_load, "Client", server.client)
    monkeypatch.setattr(late_load, "spawn", lambda *a, **k: object())
    monkeypatch.setattr(late_load, "wait_ready", lambda *a, **k: (0.0, {}))
    monkeypatch.setattr(late_load, "stop", lambda processes: None)
    args = late_load.parser().parse_args(list(argv))
    return late_load.attempt(args, through_frontend=False), server


def test_the_late_request_waits_as_long_as_asked(monkeypatch):
    result, server = late_load_with(monkeypatch, 200, "--timeout", "777")
    assert server.posts == [("http://127.0.0.1:8000", 777)] * 2  # late and next request
    assert result["first"]["status"] == 200 and result["next"]["status"] == 200


def test_a_failed_late_load_is_not_followed_by_a_second_load(monkeypatch):
    result, server = late_load_with(monkeypatch, 500, resident=False)
    assert result["first"]["status"] == 500
    assert result["next"] is None and len(server.posts) == 1


def test_a_late_load_cut_off_by_a_timeout_is_still_followed_up(monkeypatch):
    result, server = late_load_with(
        monkeypatch, 504
    )  # the frontend gave up; the worker did not
    assert result["first"]["status"] == 504 and result["next"]["status"] == 200
    result, _ = late_load_with(monkeypatch, None, error=TimeoutError("timed out"))
    assert (
        "TimeoutError" in result["first"]["status"] and result["next"]["status"] == 200
    )


def test_the_length_walk_records_why_it_stopped():
    def walk(cap, count):
        records = []
        lengths.measure(
            ["A"],
            lambda s, w: (30.0, min(w + 56, cap)),
            1,
            1,
            count,
            set(),
            records.append,
        )
        return next(r for r in records if r["type"] == "walk")

    assert walk(512, 1000) == {
        "type": "walk",
        "lengths": 456,
        "stopped": "window",
        "longest_tokens": 512,
    }
    assert walk(512, 10) == {
        "type": "walk",
        "lengths": 10,
        "stopped": "count",
        "longest_tokens": 66,
    }


def test_every_field_late_load_prints_is_explained_in_its_docstring(monkeypatch):
    import late_load

    result, _ = late_load_with(monkeypatch, 200)
    missing = [key for key in result if f"`{key}`" not in late_load.__doc__]
    assert not missing, missing


def test_late_load_refuses_a_late_checkpoint_the_worker_already_serves(monkeypatch):
    import late_load

    monkeypatch.setattr(
        late_load, "spawn", lambda *a, **k: pytest.fail("must not start")
    )
    for late in ("english", "en", "English"):
        monkeypatch.setattr(
            sys, "argv", ["late_load.py", "--model", "english", "--late", late]
        )
        with pytest.raises(SystemExit):
            late_load.main()
