"""Unit tests for the Laya worker. A fake Router stands in for laya's; no model is loaded.

python -m pytest src/models/laya/tests
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from frontend import laya_mps as worker
from models.laya import engine, optimize

ANSWER = {"type": "noul", "noul": 0.9, "confidence": 0.9}


def on_gpu(rows):
    """Stands in for input_ids on the GPU: the wrapper only reads its device type and number of rows."""
    return SimpleNamespace(device=SimpleNamespace(type="mps"), shape=(rows, 7))


class FakeAgent:
    def __init__(self, device="mps", dtype="torch.float16"):
        self.device = device
        self.dtype = dtype
        self.mps_amp_min_rows = 5


class FakeRouter:
    """The part of laya.router.Router the worker and laya.serve.create_app use."""

    def __init__(self, agent=None, fail_on_call=None, agents=None):
        self.agent = agent or FakeAgent()
        self.agents = agents if agents is not None else {"english": self.agent}
        self.calls = []
        self.loads = []
        self.hooks = []
        self.fail_on_call = fail_on_call

    @property
    def loaded(self):
        return list(self.agents)

    def load(self, name):
        self.loads.append(name)
        return self.agents.setdefault(name, self.agent)

    def add_hook(self, hook):
        self.hooks.append(hook)

    def remove_hook(self, hook):
        self.hooks.remove(hook)

    def unload(self, name):
        self.agents.pop(name, None)
        for hook in self.hooks:
            hook.on_evict(SimpleNamespace(model=name))

    def load_while_serving(self, name, agent):
        """What laya's Router.load does for a checkpoint that is not resident yet."""
        self.agents[name] = agent
        for hook in self.hooks:
            hook.on_load(SimpleNamespace(model=name, agent=agent))

    def predict(self, state, questions, model=None):
        self.calls.append((state, questions, model))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("MPS backend out of memory")
        return {
            "model": "laya-rl-agent",
            "answers": {qid: ANSWER for qid in questions},
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "routing": {"model": model, "repo": "convaiinnovations/laya"},
        }


def test_warmup_covers_short_long_and_fp16_multi_question_shapes():
    router = FakeRouter()
    engine.warmup(router, "english")
    words = {len(state.split()) for state, _, _ in router.calls}
    rows = {len(questions) for _, questions, _ in router.calls}
    assert min(words) <= 20 and max(words) >= 400
    assert max(rows) >= router.agent.mps_amp_min_rows
    assert {q["type"] for _, questions, _ in router.calls for q in questions.values()} == {"choice", "score", "noul"}
    assert len(router.calls) == len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS
    assert all(model == "english" for _, _, model in router.calls)


def test_warmup_runs_before_the_app_exists():
    router = FakeRouter()
    worker.build_app(router, "english", "mps")
    assert len(router.calls) == len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS


def test_warmup_failure_raises_and_no_app_is_built():
    with pytest.raises(RuntimeError, match="out of memory"):
        worker.build_app(FakeRouter(fail_on_call=3), "english", "mps")


def test_health_reports_the_agent_device_not_the_requested_one():
    router = FakeRouter(FakeAgent(device="cpu", dtype="torch.float32"))
    health = TestClient(worker.build_app(router, "english", "mps")).get("/health").json()
    assert health["device"] == "cpu"
    assert health["requested_device"] == "mps"
    assert health["device_mismatch"] is True
    assert health["ready"] is True


def test_health_on_the_requested_device():
    health = TestClient(worker.build_app(FakeRouter(), "english", "mps")).get("/health").json()
    assert health["device"] == "mps"
    assert health["device_mismatch"] is False
    assert health["autocast_dtype"] == "torch.float16"
    assert health["checkpoint"] == "convaiinnovations/laya"
    assert health["warmup_ms"] >= 0


def test_device_index_is_not_a_mismatch():
    router = FakeRouter(FakeAgent(device="cuda:0"))
    assert TestClient(worker.build_app(router, "english", "cuda")).get("/health").json()["device_mismatch"] is False


def test_auto_device_is_never_a_mismatch():
    router = FakeRouter(FakeAgent(device="cpu"))
    health = TestClient(worker.build_app(router, "english", None)).get("/health").json()
    assert health["requested_device"] == "auto"
    assert health["device_mismatch"] is False


def test_require_device_refuses_to_serve_on_another_device():
    router = FakeRouter(FakeAgent(device="cpu"))
    with pytest.raises(RuntimeError, match="asked for mps, english is on cpu"):
        worker.build_app(router, "english", "mps", require_device=True)


def test_only_one_health_route_remains():
    app = worker.build_app(FakeRouter(), "english", "mps")
    assert [r.path for r in app.router.routes if getattr(r, "path", None) == "/health"] == ["/health"]


def test_decisions_still_go_through_laya_serve():
    router = FakeRouter()
    client = TestClient(worker.build_app(router, "english", "mps"))
    before = len(router.calls)
    response = client.post(
        "/v1/systemone",
        json={"model": "english", "state": "refund me", "questions": {"r": {"type": "noul", "instructions": "?"}}},
    )
    assert response.status_code == 200
    assert response.json()["answers"]["r"]["noul"] == 0.9
    assert len(router.calls) == before + 1


def test_main_exits_non_zero_when_warmup_fails(monkeypatch, caplog):
    monkeypatch.setattr(worker, "make_router", lambda device, model: FakeRouter(fail_on_call=1))
    monkeypatch.setattr(sys, "argv", ["laya_mps", "--device", "mps"])
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not bind"))
    with caplog.at_level("ERROR", logger="laya-worker"), pytest.raises(SystemExit, match="not starting"):
        worker.main()
    assert "startup failed" in caplog.text and "Traceback" in caplog.text and "out of memory" in caplog.text


def test_compile_wraps_the_model_before_warmup(monkeypatch):
    router = FakeRouter()
    order = []
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: order.append((agent, len(router.calls))))
    worker.build_app(router, "english", "mps", compile=True, graph_counter=lambda: 3)
    assert order == [(router.agent, 0)]


def test_health_reports_compile_off_by_default():
    health = TestClient(worker.build_app(FakeRouter(), "english", "mps")).get("/health").json()
    assert health["compile"] == {"enabled": False}


def test_health_flags_graphs_compiled_after_ready(monkeypatch):
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: None)
    graphs = iter([4, 4, 5])  # at readiness, first /health, second /health after a new shape compiled
    client = TestClient(
        worker.build_app(FakeRouter(), "english", "mps", compile=True, graph_counter=lambda: next(graphs))
    )
    first = client.get("/health").json()["compile"]
    assert first == {
        "enabled": True,
        "active": False,  # compile_agent is stubbed out here
        "graphs_at_ready": 4,
        "graphs_now": 4,
        "recompiled_after_ready": False,
    }
    assert client.get("/health").json()["compile"]["recompiled_after_ready"] is True


def test_compile_failure_means_no_app(monkeypatch):
    def broken(agent):
        raise RuntimeError("inductor: unsupported op on mps")

    monkeypatch.setattr(optimize, "compile_agent", broken)
    with pytest.raises(RuntimeError, match="unsupported op"):
        worker.build_app(FakeRouter(), "english", "mps", compile=True)


def test_compiled_paths_by_batch_rows(monkeypatch):
    import torch

    class Encoder(torch.nn.Module):
        def forward(self, input_ids):
            return "eager encoder"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.head = torch.nn.Linear(2, 2)

        def forward(self, input_ids):
            return ("head", self.encoder(input_ids))

    class Stub(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind

        def forward(self, *args):
            return self.kind

    def fake_compile(module, dynamic):
        assert dynamic is True
        return Stub("compiled encoder" if isinstance(module, Encoder) else "whole model compiled")

    monkeypatch.setattr(torch, "compile", fake_compile)
    agent = FakeAgent()
    model = Model()
    agent.model = model
    optimize.compile_agent(agent)
    assert agent.model(on_gpu(1)) == "whole model compiled"
    assert agent.model(on_gpu(3)) == ("head", "compiled encoder")
    assert agent.model(torch.zeros(1, 7)) == ("head", "eager encoder")
    assert model.encoder(torch.zeros(1, 7)) == "eager encoder"  # the original model is left as it was
    assert len(list(agent.model.parameters())) == len(list(model.parameters()))  # one set of weights


def test_every_loaded_model_is_warmed_and_described():
    agents = {"english": FakeAgent(), "multilingual": FakeAgent(device="cpu", dtype="torch.float32")}
    router = FakeRouter(agents=agents)
    health = TestClient(worker.build_app(router, "english", "mps")).get("/health").json()
    per_model = len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS
    assert [m for _, _, m in router.calls].count("multilingual") == per_model
    assert [m for _, _, m in router.calls].count("english") == per_model
    assert set(health["models"]) == {"english", "multilingual"}
    assert health["device"] == "mps"  # the top level summarises --model
    assert health["models"]["multilingual"]["device"] == "cpu"
    assert health["device_mismatch"] is True


def test_a_model_that_is_not_preloaded_is_not_loaded_for_warmup():
    router = FakeRouter(agents={"multilingual": FakeAgent()})
    health = TestClient(worker.build_app(router, "english", "mps")).get("/health").json()
    assert "english" not in router.loads
    assert {m for _, _, m in router.calls} == {"multilingual"}
    assert set(health["models"]) == {"multilingual"}


def test_nothing_preloaded_warms_the_worker_model():
    router = FakeRouter(agents={})
    worker.build_app(router, "english", "mps")
    assert {m for _, _, m in router.calls} == {"english"}


def test_revision_comes_from_the_loaded_snapshot_not_a_guess():
    revisions = {"convaiinnovations/laya": "55cf4c4"}
    health = TestClient(worker.build_app(FakeRouter(), "english", "mps", revisions=revisions)).get("/health")
    assert health.json()["revision"] == "55cf4c4"
    unknown = TestClient(worker.build_app(FakeRouter(), "english", "mps")).get("/health").json()
    assert unknown["revision"] is None


def test_record_snapshot_revisions_reads_the_downloaded_path(monkeypatch):
    import huggingface_hub

    paths = {
        "convaiinnovations/laya": "/cache/models--convaiinnovations--laya/snapshots/55cf4c4abc/multilingual",
        "/local/checkpoint": "/local/checkpoint",
    }
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **kwargs: paths[repo_id])
    revisions = engine.record_snapshot_revisions()
    assert (
        huggingface_hub.snapshot_download("convaiinnovations/laya", allow_patterns=["*"])
        == paths["convaiinnovations/laya"]
    )
    huggingface_hub.snapshot_download("/local/checkpoint")
    assert revisions == {"convaiinnovations/laya": "55cf4c4abc"}
    huggingface_hub.snapshot_download(
        "convaiinnovations/laya", allow_patterns=["multilingual/model.safetensors", "multilingual/tokenizer/*"]
    )
    huggingface_hub.snapshot_download("convaiinnovations/laya", allow_patterns=["model.safetensors", "tokenizer/*"])
    assert set(revisions) == {"convaiinnovations/laya", "convaiinnovations/laya/multilingual"}


def test_fp16_weights_keep_act_head_in_fp32():
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.act_head = torch.nn.Linear(4, 2)

    agent = FakeAgent()
    agent.model = Model()
    optimize.use_fp16_weights(agent)
    assert agent.model.eager.encoder.weight.dtype == torch.float16
    assert agent.model.eager.act_head.weight.dtype == torch.float32


def test_fp16_weights_are_applied_to_every_loaded_model_before_warmup(monkeypatch):
    order = []
    agents = {"english": FakeAgent(), "multilingual": FakeAgent()}
    router = FakeRouter(agents=agents)
    monkeypatch.setattr(optimize, "use_fp16_weights", lambda agent: order.append((agent, len(router.calls))))
    worker.build_app(router, "english", "mps", fp16=True)
    assert order == [(agents["english"], 0), (agents["multilingual"], 0)]


def test_options_are_not_applied_to_a_model_on_the_cpu(monkeypatch, caplog):
    applied = []
    monkeypatch.setattr(optimize, "use_fp16_weights", lambda agent: applied.append("fp16"))
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: applied.append("compile"))
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(
            FakeRouter(FakeAgent(device="cpu")), "english", "cpu", fp16=True, compile=True, graph_counter=lambda: 0
        )
    assert applied == []
    assert "apply on the GPU only" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(FakeRouter(), "english", "mps", fp16=True, compile=True, graph_counter=lambda: 0)
    assert applied == ["fp16", "compile"]
    assert "GPU only" not in caplog.text


def test_after_a_fallback_to_cpu_the_model_runs_fp32_and_uncompiled(monkeypatch):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.act_head = torch.nn.Linear(4, 2)

        def forward(self, input_ids):
            return ("eager", self.encoder.weight.dtype)

    monkeypatch.setattr(torch, "compile", lambda module, dynamic: lambda *a, **k: "compiled")
    agent = FakeAgent()
    agent.model = Model()
    optimize.use_fp16_weights(agent)
    optimize.compile_agent(agent)
    assert agent.model(on_gpu(1)) == "compiled"
    assert next(agent.model.parameters()).dtype == torch.float16
    assert agent.model(torch.zeros(1, 7)) == ("eager", torch.float32)
    assert {p.dtype for p in agent.model.parameters()} == {torch.float32}
    assert agent.model(torch.zeros(3, 7)) == ("eager", torch.float32)


def test_health_follows_a_fallback_to_cpu_after_startup():
    agent = FakeAgent(device="mps")
    client = TestClient(worker.build_app(FakeRouter(agent), "english", "mps", require_device=True))
    assert client.get("/health").json()["device_mismatch"] is False
    agent.device = "cpu"
    agent.dtype = "torch.float32"
    health = client.get("/health").json()
    assert health["device"] == "cpu"
    assert health["device_mismatch"] is True
    assert health["models"]["english"]["device"] == "cpu"
    assert health["autocast_dtype"] == "torch.float32"


def test_health_compile_active_follows_a_fallback_to_cpu(monkeypatch):
    import torch

    compiles = []
    monkeypatch.setattr(torch, "compile", lambda module, dynamic: compiles.append(module) or module)
    agent = FakeAgent()
    agent.model = torch.nn.Sequential()
    agent.model.encoder = torch.nn.Identity()
    client = TestClient(worker.build_app(FakeRouter(agent), "english", "mps", compile=True, graph_counter=lambda: 2))
    assert client.get("/health").json()["compile"]["active"] is True
    optimize.compile_agent(agent)  # a second name for the same agent must not compile again
    assert len(compiles) == 2
    agent.device = "cpu"
    assert client.get("/health").json()["compile"]["active"] is False


def test_warning_when_the_warmup_does_not_reach_layas_autocast_rows(caplog):
    agent = FakeAgent()
    agent.mps_amp_min_rows = engine.WARMUP_MAX_ROWS + 1
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(FakeRouter(agent), "english", "mps")
    assert "the warmup stops at" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(FakeRouter(), "english", "mps")
    assert "the warmup stops at" not in caplog.text


def test_log_level_applies_to_the_workers_own_log(monkeypatch):
    seen = {}
    monkeypatch.setattr(worker, "make_router", lambda device, model: FakeRouter())
    monkeypatch.setattr(worker.logging, "basicConfig", lambda **kw: seen.update(kw))
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(uvicorn=kw))
    monkeypatch.setattr(sys, "argv", ["laya_mps", "--device", "mps", "--log-level", "warning"])
    worker.main()
    assert (seen["level"], seen["uvicorn"]["log_level"]) == ("WARNING", "warning")
    assert (seen["uvicorn"]["host"], seen["uvicorn"]["port"]) == ("127.0.0.1", 8000)  # local only by default


def test_a_checkpoint_loaded_while_serving_is_prepared_and_described(monkeypatch):
    applied = []
    monkeypatch.setattr(optimize, "use_fp16_weights", lambda agent: applied.append(agent))
    router = FakeRouter()
    client = TestClient(worker.build_app(router, "english", "mps", fp16=True))
    before = len(router.calls)
    late = FakeAgent(device="cpu", dtype="torch.float32")
    router.load_while_serving("multilingual", late)
    assert applied == [router.agent]  # the late one is on the CPU, where the options do not apply
    assert [m for _, _, m in router.calls[before:]] == ["multilingual"] * (
        len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS
    )
    health = client.get("/health").json()
    assert set(health["models"]) == {"english", "multilingual"}
    assert health["models"]["multilingual"]["device"] == "cpu"
    assert health["device_mismatch"] is True
    assert health["preparing"] == []
    router.unload("multilingual")
    health = client.get("/health").json()
    assert set(health["models"]) == {"english"}
    assert health["device_mismatch"] is False


def test_a_late_checkpoint_that_cannot_be_prepared_is_not_kept():
    router = FakeRouter()
    client = TestClient(worker.build_app(router, "english", "mps", require_device=True))
    with pytest.raises(RuntimeError, match="multilingual is on cpu"):
        router.load_while_serving("multilingual", FakeAgent(device="cpu"))
    assert router.loaded == ["english"]
    assert set(client.get("/health").json()["models"]) == {"english"}
    router.fail_on_call = len(router.calls) + 1
    with pytest.raises(RuntimeError, match="out of memory"):
        router.load_while_serving("multilingual", FakeAgent())
    assert router.loaded == ["english"]


def test_compile_baseline_moves_when_a_late_checkpoint_compiles(monkeypatch):
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: None)
    graphs = iter([3, 7, 7])  # startup, after the late checkpoint compiled, /health
    router = FakeRouter()
    client = TestClient(worker.build_app(router, "english", "mps", compile=True, graph_counter=lambda: next(graphs)))
    router.load_while_serving("multilingual", FakeAgent())
    compiled = client.get("/health").json()["compile"]
    assert (compiled["graphs_at_ready"], compiled["recompiled_after_ready"]) == (7, False)


def test_startup_error_names_every_checkpoint_off_the_requested_device():
    agents = {"english": FakeAgent(device="cpu"), "multilingual": FakeAgent(device="cpu")}
    with pytest.raises(RuntimeError, match="asked for mps, english is on cpu, multilingual is on cpu"):
        worker.build_app(FakeRouter(agents=agents), "english", "mps", require_device=True)


def test_health_names_a_checkpoint_while_it_is_being_prepared():
    router = FakeRouter()
    client = TestClient(worker.build_app(router, "english", "mps"))
    seen = []
    predict = router.predict

    def predict_and_look(state, questions, model=None):
        seen.append(client.get("/health").json()["preparing"])
        return predict(state, questions, model=model)

    router.predict = predict_and_look
    router.load_while_serving("multilingual", FakeAgent())
    assert seen[0] == ["multilingual"]
    assert client.get("/health").json()["preparing"] == []


class StubAgent:
    """Stands in for laya.agent.Agent under laya's real Router: no weights, fixed answers."""

    devices: dict = {}  # subfolder -> the device that checkpoint lands on

    def __init__(self, repo, device=None, token=None, subfolder=None):
        self.device = self.devices.get(subfolder, device)
        self.dtype = "torch.float16"
        self.mps_amp_min_rows = 5

    def system_one(self, state, questions, lang=None, **_):
        return {"model": "stub", "answers": {qid: ANSWER for qid in questions}, "usage": {}}


def test_a_failed_late_load_gives_back_the_checkpoint_laya_evicted_for_it(monkeypatch):
    import laya.agent
    from laya.router import Router

    monkeypatch.setattr(laya.agent, "Agent", StubAgent)
    monkeypatch.setattr(StubAgent, "devices", {"typed-decisions": "cpu"})
    router = Router(device="mps")
    router.preload(["english", "multilingual"])  # laya keeps two checkpoints by default
    client = TestClient(worker.build_app(router, "english", "mps", require_device=True))
    with pytest.raises(RuntimeError, match="typed-decisions is on cpu"):
        router.predict("refund me", {"r": {"type": "noul", "instructions": "?"}}, model="typed-decisions")
    assert sorted(router.loaded) == ["english", "multilingual"]
    health = client.get("/health").json()
    assert set(health["models"]) == {"english", "multilingual"}
    assert health["preparing"] == []


def test_building_a_second_app_on_a_router_replaces_the_first_apps_hooks():
    router = FakeRouter()
    worker.build_app(router, "english", "mps")
    worker.build_app(router, "english", "mps")
    assert len(router.hooks) == 1
    before = len(router.calls)
    router.load_while_serving("multilingual", FakeAgent())
    assert len(router.calls) - before == len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS


def test_main_warns_about_laya_serve_variables_it_does_not_read(monkeypatch, caplog):
    monkeypatch.setattr(worker, "make_router", lambda device, model: FakeRouter())
    monkeypatch.setattr(worker.logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: None)
    monkeypatch.setattr(sys, "argv", ["laya_mps", "--device", "mps"])
    monkeypatch.setenv("LAYA_THREADS", "4")
    monkeypatch.setenv("LAYA_API_KEY", "k")
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.main()
    assert "LAYA_THREADS" in caplog.text
    assert "LAYA_API_KEY" not in caplog.text


def test_graphs_compiled_by_a_late_checkpoint_that_fails_are_not_reported_as_recompiles(monkeypatch):
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: None)
    router = FakeRouter()
    graphs = {"n": 0}
    predict = router.predict

    def predict_and_compile(state, questions, model=None):
        graphs["n"] += 1
        return predict(state, questions, model=model)

    router.predict = predict_and_compile
    client = TestClient(worker.build_app(router, "english", "mps", compile=True, graph_counter=lambda: graphs["n"]))
    router.fail_on_call = len(router.calls) + 3
    with pytest.raises(RuntimeError, match="out of memory"):
        router.load_while_serving("multilingual", FakeAgent())
    assert client.get("/health").json()["compile"]["recompiled_after_ready"] is False


class Repository:
    """What the downloading stub agents see: the repository's current commit, and where checkpoints land."""

    commit = 0
    device: dict = {}  # subfolder -> device
    broken: set = set()  # subfolders whose forward raises


class DownloadingAgent:
    """laya.agent.Agent without weights. It downloads the way laya does, so the worker's recording sees it."""

    def __init__(self, repo, device=None, token=None, subfolder=None):
        import huggingface_hub

        prefix = f"{subfolder}/" if subfolder else ""
        path = huggingface_hub.snapshot_download(
            repo, token=token, allow_patterns=[prefix + "model.safetensors", prefix + "tokenizer/*"]
        )
        self.loaded_commit = Path(path).name
        self.subfolder = subfolder
        self.device = Repository.device.get(subfolder, device)
        self.dtype = "torch.float16"
        self.mps_amp_min_rows = 5

    def system_one(self, state, questions, lang=None, **_):
        if self.subfolder in Repository.broken:
            raise RuntimeError("MPS backend out of memory")
        return {"model": "stub", "answers": {qid: ANSWER for qid in questions}, "usage": {}}


@pytest.fixture
def laya_router(monkeypatch):
    """laya's real Router over downloading stub agents, and the revisions the worker records for them."""
    import huggingface_hub
    import laya.agent
    from laya.router import Router

    monkeypatch.setattr(Repository, "commit", 0)
    monkeypatch.setattr(Repository, "device", {})
    monkeypatch.setattr(Repository, "broken", set())
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: f"/hf/snapshots/commit-{Repository.commit}"
    )
    monkeypatch.setattr(laya.agent, "Agent", DownloadingAgent)
    return Router(device="mps"), engine.record_snapshot_revisions()


def assert_health_matches(client, router):
    health = client.get("/health").json()
    agents = dict(router._agents)
    assert set(health["models"]) == set(router.loaded) == set(agents)
    assert health["preparing"] == []
    for name, agent in agents.items():
        assert health["models"][name]["device"] == str(agent.device)
        assert health["models"][name]["revision"] == agent.loaded_commit
    assert health["device_mismatch"] == any(str(agent.device) != "mps" for agent in agents.values())


def test_checkpoints_of_one_repository_keep_the_revision_of_their_own_load(laya_router):
    router, revisions = laya_router
    router.load("english")
    Repository.commit = 1  # the repository moves on between the two loads
    router.load("multilingual")
    client = TestClient(worker.build_app(router, "english", "mps", revisions=revisions))
    assert_health_matches(client, router)
    Repository.commit = 2
    router.predict("refund me", {"r": {"type": "noul", "instructions": "?"}}, model="typed-decisions")
    assert_health_matches(client, router)


@pytest.mark.parametrize("require_device", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_health_matches_the_router_after_any_sequence_of_loads(laya_router, seed, require_device):
    import random

    router, revisions = laya_router
    names = ["english", "multilingual", "typed-decisions"]
    subfolder = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
    rng = random.Random(seed)
    router.preload(rng.sample(names, rng.choice([1, 2])))
    client = TestClient(
        worker.build_app(router, router.loaded[0], "mps", require_device=require_device, revisions=revisions)
    )
    assert_health_matches(client, router)
    for _ in range(12):
        Repository.commit += rng.random() < 0.3
        Repository.device = {subfolder[n]: "cpu" for n in names if rng.random() < 0.25}
        Repository.broken = {subfolder[n] for n in names if rng.random() < 0.15}
        try:
            router.predict("refund me", {"r": {"type": "noul", "instructions": "?"}}, model=rng.choice(names))
        except RuntimeError:
            pass  # a failed late load or forward: the request fails, /health must still match the router
        Repository.broken = set()
        assert_health_matches(client, router)


def test_health_answers_with_no_resident_checkpoint(laya_router):
    router, revisions = laya_router
    router.preload(["english"])
    client = TestClient(worker.build_app(router, "english", "mps", revisions=revisions))
    router.unload()
    assert client.get("/health").json()["models"] == {}


def test_a_checkpoint_loaded_while_serving_gets_the_gpu_options_before_its_warmup(monkeypatch):
    applied = []
    router = FakeRouter()
    monkeypatch.setattr(optimize, "use_fp16_weights", lambda agent: applied.append(("fp16", len(router.calls))))
    monkeypatch.setattr(optimize, "compile_agent", lambda agent: applied.append(("compile", len(router.calls))))
    worker.build_app(router, "english", "mps", fp16=True, compile=True, graph_counter=lambda: 0)
    warmed_at_startup = len(router.calls)
    router.load_while_serving("multilingual", FakeAgent())
    assert applied[2:] == [("fp16", warmed_at_startup), ("compile", warmed_at_startup)]


def test_a_checkpoint_off_the_requested_device_is_logged_when_it_is_allowed(caplog):
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(FakeRouter(FakeAgent(device="cpu")), "english", "mps")
    assert "asked for mps, english is on cpu" in caplog.text


def test_no_warning_when_the_warmup_just_reaches_layas_autocast_rows(caplog):
    agent = FakeAgent()
    agent.mps_amp_min_rows = engine.WARMUP_MAX_ROWS
    with caplog.at_level("WARNING", logger="laya-worker"):
        worker.build_app(FakeRouter(agent), "english", "mps")
    assert "the warmup stops at" not in caplog.text


def test_failed_late_loads_are_logged_with_their_cause(laya_router, caplog):
    router, revisions = laya_router
    router.preload(["english", "multilingual"])
    worker.build_app(router, "english", "mps", require_device=True, revisions=revisions)
    Repository.device = {"typed-decisions": "cpu", None: "cpu"}  # english cannot come back either
    with caplog.at_level("ERROR", logger="laya-worker"), pytest.raises(RuntimeError):
        router.predict("refund me", {"r": {"type": "noul", "instructions": "?"}}, model="typed-decisions")
    assert "typed-decisions could not be prepared and is unloaded" in caplog.text
    assert "english was evicted for typed-decisions and could not be loaded again" in caplog.text
    assert "asked for mps, typed-decisions is on cpu" in caplog.text  # the traceback of the cause
    assert router.loaded == ["multilingual"]


def test_a_failed_late_load_reloads_only_what_was_evicted_for_it(laya_router, monkeypatch):
    router, revisions = laya_router
    router.max_loaded = 1
    router.load("english")
    worker.build_app(router, "english", "mps", require_device=True, revisions=revisions)
    router.predict("refund me", {"r": {"type": "noul", "instructions": "?"}}, model="multilingual")  # evicts english
    built = []
    original = DownloadingAgent.__init__
    monkeypatch.setattr(
        DownloadingAgent,
        "__init__",
        lambda self, repo, **kw: built.append(kw.get("subfolder")) or original(self, repo, **kw),
    )
    Repository.device = {"typed-decisions": "cpu"}
    with pytest.raises(RuntimeError):
        router.predict("refund me", {"r": {"type": "noul", "instructions": "?"}}, model="typed-decisions")
    assert built == ["typed-decisions", "multilingual"] and router.loaded == ["multilingual"]


def test_describe_reads_the_weight_dtype_from_the_model_as_it_is():
    import torch

    agent = FakeAgent()
    assert engine.describe(agent, "mps", None)["weights_dtype"] is None  # no model to read
    agent.model = torch.nn.Linear(2, 2)
    assert engine.describe(agent, "mps", None)["weights_dtype"] == "torch.float32"
    agent.model.half()
    assert engine.describe(agent, "mps", None)["weights_dtype"] == "torch.float16"


def test_warmup_time_is_reported_in_milliseconds(monkeypatch):
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(engine.time, "perf_counter", lambda: next(clock))
    assert engine.warmup(FakeRouter(), "english")["warmup_ms"] == 250.0


def test_apply_says_whether_the_options_were_applied(monkeypatch):
    monkeypatch.setattr(optimize, "use_fp16_weights", lambda agent: None)
    assert optimize.apply(FakeAgent(device="cpu"), fp16=True, compile=False) is False
    assert optimize.apply(FakeAgent(device="mps"), fp16=True, compile=False) is True


def test_the_checkpoint_is_built_once_and_serves_every_request(laya_router, monkeypatch):
    router, revisions = laya_router
    router.preload(["english"])
    built = []
    original = DownloadingAgent.__init__
    monkeypatch.setattr(
        DownloadingAgent, "__init__", lambda self, repo, **kw: built.append(repo) or original(self, repo, **kw)
    )
    client = TestClient(worker.build_app(router, "english", "mps", revisions=revisions))
    body = {"model": "english", "state": "refund me", "questions": {"r": {"type": "noul", "instructions": "?"}}}
    for _ in range(3):
        response = client.post("/v1/systemone", json=body)
        assert response.status_code == 200 and response.json()["answers"]["r"] == ANSWER
    assert built == [] and router.loaded == ["english"]


def test_main_binds_only_after_every_warmup_request(monkeypatch):
    router = FakeRouter()
    bound_after = []
    monkeypatch.setattr(worker, "make_router", lambda device, model: router)
    monkeypatch.setattr(worker.logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: bound_after.append(len(router.calls)))
    monkeypatch.setattr(sys, "argv", ["laya_mps", "--device", "mps"])
    worker.main()
    assert bound_after == [len(engine.WARMUP_SHAPES) * engine.WARMUP_REPEATS]


def test_the_warmup_asks_every_question_type():
    kinds = {q["type"] for _, questions in engine.WARMUP_SHAPES for q in questions.values()}
    assert kinds == {"choice", "score", "noul"}
