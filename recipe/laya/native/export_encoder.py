"""Export official Laya intermediates for the Rust encoder integration test."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import torch
from laya import Agent
from laya.common import collate_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert importlib.metadata.version("laya") == "0.3.20"
    agent = Agent(args.checkpoint, device="cuda", fast=False, compile=False)
    assert agent.accelerate(use_graphs=False, strict=True)
    fast = agent._fast
    assert fast is not None and not fast.use_graphs
    cases = json.loads(args.requests.read_text())
    # Exercise the largest supported batch with real, independently packed questions.
    widest = copy.deepcopy(next(c for c in cases if c["name"] == "long_3"))
    widest["name"] = "long_16"
    question = next(iter(widest["request"]["questions"].values()))
    widest["request"]["questions"] = {f"q{i}": copy.deepcopy(question) for i in range(16)}
    cases.append(widest)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        name, request = case["name"], case["request"]
        questions = request["questions"]
        internal = {k: agent._to_internal(v) for k, v in questions.items()}
        items = agent._encode_state(request["state"], list(questions), internal)
        packed = collate_items([items], agent.tok.pad_token_id)
        n, length = packed["input_ids"].shape
        b, l = 1 << (n - 1).bit_length(), (length + 15) // 16 * 16
        assert 1 <= b <= 16 and 16 <= l <= 512
        ids = torch.zeros((b, l), dtype=torch.int64, device="cuda")
        lens = torch.zeros(b, dtype=torch.int32, device="cuda")
        types = torch.zeros(b, dtype=torch.int64, device="cuda")
        ids[:n, :length] = packed["input_ids"].cuda()
        lens[:n] = packed["attention_mask"].sum(-1).to("cuda", torch.int32)
        types[:n] = packed["qtype"].cuda()

        def save(stage, value):
            suffix = f"-{stage}" if stage else ""
            (args.output / f"{name}{suffix}.f32").write_bytes(value.float().contiguous().cpu().numpy().tobytes())

        original_ln = fast.k_addln
        original_head_norm, original_ffn2 = fast.k_ln_b, fast.k_ffn2
        head_state, head_count = [None], [0]

        def head_norm(*values):
            head_state[0] = values[0]
            original_head_norm(*values)

        def ffn2(*values):
            original_ffn2(*values)
            save(f"head{head_count[0]}", head_state[0] + values[-1].float())
            head_count[0] += 1
        count = [0]

        def addln(*values):
            original_ln(*values)
            count[0] += 1
            if count[0] in (2, 4, 6, 56):
                save(f"encoder{count[0] // 2 - 1}", values[0])

        fast.k_addln = addln
        fast.k_ln_b, fast.k_ffn2 = head_norm, ffn2
        with torch.no_grad():
            embedding = torch.nn.functional.embedding(ids, fast.emb_w).reshape(-1, 1024).float()
            save("embedding", torch.nn.functional.layer_norm(embedding, (1024,), fast.emb_ln, None, fast.eps))
            hidden = fast._encode(ids, lens, types)
            save("", hidden)
        fast.k_addln = original_ln
        fast.k_ln_b, fast.k_ffn2 = original_head_norm, original_ffn2
        records.append({"name": name, "batch": b, "sequence": l,
                        "ids": ids.flatten().cpu().tolist(), "lengths": lens.cpu().tolist(),
                        "types": types.cpu().tolist()})
        print("REFERENCE", name, b, l, flush=True)
    (args.output / "cases.json").write_text(json.dumps(records))


if __name__ == "__main__":
    main()
