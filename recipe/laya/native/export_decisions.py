"""CPU decode references from Laya 0.3.20; synthetic logits, no model loading."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import laya.agent
import laya.common
import torch


def question(qid, kind, criteria=None):
    return {"id": qid, "kind": kind, "criteria": criteria}


def case(name, questions, logits, action_logits, config=None):
    row = dict(name=name, questions=questions, logits=logits, action_logits=action_logits)
    if config is not None:
        row["config"] = config
    return row


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if importlib.metadata.version("laya") != "0.3.20":
    raise SystemExit("the reference requires laya==0.3.20")
torch.set_num_threads(1)
cfg_file = args.checkpoint / "rl_agent_config.json"
cfg = json.loads(cfg_file.read_text())
config = {key: cfg[key] for key in ("temperature", "temperature_by_options")}
cases = []
for k in (1, 2, 3, 5, 6, 10, 11):
    criteria = {f"option-{i}": None for i in range(k)}
    cases.append(case(
        f"choice-{k}", [question("q", "choice", criteria)],
        [[i * 0.375 - 1.0 for i in range(k)]], [[-0.75, 0.5]],
    ))
cases += [
    case("choice-first-tie", [question("q", "choice", {"z": None, "a": None})],
         [[2.0, 2.0]], [[0.0, 0.0]]),
    case("score-single", [question("q", "score", ["only"])], [[-5.0]], [[1.0, -1.0]]),
    case("score-legend", [question("q", "score", ["low", {"description": "middle"}, 7])],
         [[0.25, 1.5, -0.5]], [[0.3, -0.8]]),
    case("noul-extremes", [question("false", "noul"), question("true", "noul")],
         [[1e30, -1e30], [-1e30, 1e30]], [[1e30, -1e30], [-1e30, 1e30]]),
    case("temperature-clamps", [question("choice", "choice", {"a": None, "b": None}),
                               question("score", "score", ["low", "mid", "high"])],
         [[-0.4, 0.6], [-0.4, 0.6, 1.6]], [[-1.0, 1.0], [2.0, -2.0]],
         {"temperature": [0.01, 80.0, 1.0], "temperature_by_options": {}}),
    case("mixed-order", [question("z", "score", ["low", "mid", "high"]),
                         question("a", "choice", {"later": "", "earlier": ""}),
                         question("m", "noul")],
         [[-0.5, 1.0, 0.75], [0.5, -0.5], [0.0, 0.0]],
         [[0.0, 1.0], [2.0, -1.0], [-2.0, 0.0]]),
    case("score-bucket-override", [question("q", "score", list(range(6)))],
         [[-2.0, -1.0, 0.0, 0.5, 1.0, 3.0]], [[-0.25, 0.75]],
         {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {"score:6-10": 4.0}}),
    case("rounding-boundaries", [question(q, "choice", {"a": None, "b": None})
                                 for q in ("prob-low", "prob-high", "entropy-low", "entropy-high")],
         [[math.log(p / (1 - p)), 0.0] for p in (0.800049, 0.800051)]
         + [[1.3862165, 0.0], [1.3862353, 0.0]], [[-0.25, 0.75]] * 4,
         {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}),
    case("empty", [], [], []),
]
rounding_probe = case(
    "fp32-reduction-boundary", [question("q", "choice", {str(i): None for i in range(16)})],
    [[-1.7411574125289917, -0.19089631736278534, -0.6029739379882812, -0.8184939026832581,
      0.16066476702690125, -0.4026077389717102, 0.343989759683609, -0.600969135761261,
      0.8842262029647827, -0.26977965235710144, -0.7890094518661499, 0.2582162916660309,
      0.85430908203125, -0.11924569308757782, 0.9091809988021851, -0.00020837262854911387]],
    [[0.0, 0.0]], {"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}},
)
for row in cases + [rounding_probe]:
    current = row.get("config", config)
    agent = laya.agent.Agent.__new__(laya.agent.Agent)
    agent.temperature = [laya.common.clamp_temperature(t) for t in current["temperature"]]
    agent.temperature_by_options = {
        key: laya.common.clamp_temperature(t) for key, t in current["temperature_by_options"].items()
    }
    agent.lang_temperatures = {}
    logits = torch.full((len(row["logits"]), max(map(len, row["logits"]), default=0)), -1e4)
    for i, values in enumerate(row["logits"]):
        logits[i, :len(values)] = torch.tensor(values, dtype=torch.float32)
    actions = torch.tensor(row["action_logits"], dtype=torch.float32).reshape(-1, 2)
    agent._infer = lambda batch: (logits, actions)
    logits_np, action_probs = agent._forward(None)
    internal = {q["id"]: {"t": q["kind"], "crit": q["criteria"]} for q in row["questions"]}
    items = [{"markers": list(range(len(values)))} for values in row["logits"]]
    row["answers"] = agent._decode_answers(logits_np, action_probs, items, list(internal), internal, 0)

reference = {
    "python": platform.python_version(),
    **{name: importlib.metadata.version(name) for name in ("laya", "torch", "numpy")},
    "source_sha256": {
        Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (laya.agent, laya.common)
    },
    "config_sha256": hashlib.sha256(cfg_file.read_bytes()).hexdigest(),
    "scope": "Synthetic boundary logits; official CPU decode parity, not model quality or performance.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
dump = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
args.output.write_text(
    '{"reference":' + dump(reference) + ',"config":' + dump(config) + ',"cases":[\n'
    + ",\n".join(dump(row) for row in cases) + '\n],"rounding_probe":' + dump(rounding_probe) + "}\n"
)
print(f"{len(cases)} exact cases and one rounding probe written to {args.output}")
