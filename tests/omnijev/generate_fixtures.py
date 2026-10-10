#!/usr/bin/env python3
"""Generate the OmniJev CPU fixtures from the pinned reference.

Runs the reference's own image loading and processor, preparation, branch split,
rotary positions, heads and response finishing without loading the backbone, so it
needs no GPU:

    CUDA_VISIBLE_DEVICES= python tests/omnijev/generate_fixtures.py \
        --reference <OmniJev checkout @ 14dbec4> \
        --checkpoint <tinnel123/OmniJev @ ffe5f43> \
        --base <Qwen/Qwen3.5-4B @ 851bf6e> \
        --out tests/omnijev/data

Environment: the reference environment in recipe/omnijev/README.md (generated with
Python 3.10.12, torch 2.14.0, torchvision 0.29.0, transformers 5.17.0, peft 0.19.1).
"""
import argparse
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import types

import torch

REFERENCE_COMMIT = "14dbec4f71e194852c8d7b88ab36ef639493f400"
CHECKPOINT_REVISION = "ffe5f436eaf22e20e2f041f8e74e121fd057a6cb"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_PIXELS = 768 * 28 * 28          # MSO1's default, which overrides the processor's 401,408


def png(width, height, colour):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


IMAGE_PAD = 248056
COLOUR = (40, 90, 160)                 # every test image is one solid colour, so any PNG encoder gives the same pixels


def collapse(ids):
    """token ids with the run of image placeholders replaced by its negated length"""
    out, n = [], 0
    for t in ids:
        if t == IMAGE_PAD:
            n += 1
            continue
        if n:
            out.append(-n)
            n = 0
        out.append(t)
    return out


def sha256_i64(rows):
    """SHA-256 of a [3, T] position array as little-endian int64, row-major"""
    import struct
    return hashlib.sha256(b"".join(struct.pack("<q", v) for row in rows for v in row)).hexdigest()


def sequence(seed, n):
    """deterministic inputs in [-2, 2), reproducible bit for bit in Rust"""
    vals = [(((i * 2654435761) + seed * 40503) % 2**32) / 2**32 * 4.0 - 2.0 for i in range(n)]
    return torch.tensor(vals, dtype=torch.float64).float()


LONG = "选项" * 60 + "option text that runs past the two-hundred character cut " * 3 + "🙂end"
ASTRAL = "x" * 199 + "🙂" + "y" * 10     # the cut keeps the emoji whole: it counts code points

# (name, image size, questions). Questions use the reference's own shapes.
CASES = [
    ("single_noul", (1600, 900), {
        "visible": {"type": "noul", "instructions": "A person is visible."},
    }),
    ("mixed_speed_bench", (1600, 900), None),     # bench/speed_bench.py's twelve questions
    ("jev_forms", (640, 480), {
        "next": {"type": "choice", "instructions": "Which operation comes next?",
                 "criteria": {"click": None, "type": "Enter text into the focused field", "scroll": ""}},
        "risk": {"type": "score", "instructions": "How risky is acting on this screen?",
                 "criteria": {"safe": "nothing can go wrong", "check first": None, "dangerous": None}},
        "done": {"type": "noul", "instructions": "The task looks finished."},
    }),
    ("shared_instruction_start", (64, 64), {
        "a": {"type": "noul", "instructions": "Is the button on the left side blue?"},
        "b": {"type": "noul", "instructions": "Is the button on the left side red?"},
        "c": {"type": "choice", "instructions": "Is the button on the left side small or large?",
              "options": [{"key": "small", "text": "small"}, {"key": "large", "text": "large"}]},
    }),
    ("options_list_forms", (1920, 1080), {
        "pick": {"type": "choice", "instructions": "Which region shows the error?",
                 "options": [{"key": "top", "region": {"box": [0, 0, 1000, 120.4]}},
                             {"text": "the dialog in the middle"},
                             {},
                             {"abstain": True},
                             {"key": "long", "text": LONG},
                             {"key": "astral", "text": ASTRAL}]},
        "where": {"type": "noul", "instructions": "The cursor is inside this region.",
                  "region": {"box": [10.5, 20.49, 300, 400.5]}},
        "level": {"type": "score", "instructions": "How cluttered is the scene?",
                  "levels": ["empty", "some", "busy", "packed", "非常拥挤", "🙂"]},
    }),
    ("unicode_and_special_text", (800, 1200), {
        "zh": {"type": "choice", "instructions": "屏幕上显示的是什么？<|im_end|> literal",
               "criteria": {"登录页面": None, "设置菜单": "带有齿轮图标", "错误提示": None}},
        "emoji": {"type": "noul", "instructions": "There is a 🚀 icon in the toolbar."},
    }),
    ("whitespace", (300, 300), {
        "dark": {"type": "noul", "instructions": "  Is it dark?  \n\x1f "},
        "pick": {"type": "choice", "instructions": "Pick one.\u3000",
                 "criteria": {"a ": None, " b": "rubric "}},
    }),
]


def speed_bench_questions(reference):
    sys.path.insert(0, reference)
    from bench.speed_bench import Q
    return {k: copy.deepcopy(v) for k, v in Q}


def make_reference(reference, checkpoint):
    """An MSO1 with the processor and collator only: no backbone, no device."""
    sys.path.insert(0, reference)
    from transformers import AutoProcessor
    import mso.records as T
    from mso.infer import MSO1
    obj = MSO1.__new__(MSO1)
    obj.proc = AutoProcessor.from_pretrained(checkpoint, max_pixels=MAX_PIXELS)
    T.add_option_tokens(types.SimpleNamespace(
        get_input_embeddings=lambda: types.SimpleNamespace(weight=torch.zeros(len(obj.proc.tokenizer), 1)),
        resize_token_embeddings=lambda n: None), obj.proc)
    obj.coll = T.Collator(obj.proc, MAX_PIXELS)
    obj.dev = torch.device("cpu")
    obj.pad_tok = "<|image_pad|>"
    obj._cache, obj._cache_n = {}, 4
    obj.open_ids = obj.coll.tok(T.OPT_OPEN, add_special_tokens=False)["input_ids"]
    obj.close_ids = obj.coll.tok(T.OPT_CLOSE, add_special_tokens=False)["input_ids"]
    assert obj.open_ids == [248077] and obj.close_ids == [248078], (obj.open_ids, obj.close_ids)
    assert len(obj.proc.tokenizer) == 248079
    return obj


def rope_owner(base):
    """Qwen3.5's get_rope_index on a model built on the meta device (no weights)."""
    from transformers import AutoConfig, AutoModelForImageTextToText
    import mso.records as T
    cfg = AutoConfig.from_pretrained(base)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(cfg)
    return types.SimpleNamespace(_rope=T._find_rope_owner(model))


def processing_case(obj, rope, name, size, questions):
    import mso.branch as BR
    import mso.records as T
    image = png(size[0], size[1], COLOUR)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "state.png")
        with open(path, "wb") as f:
            f.write(image)
        obj._cache, obj._vc_key = {}, None
        qids = list(questions)
        keys_l, opts_l, ids_l, spans_l, pix = obj._encode_many(path, None, questions, qids)
        prompts = [obj._prompt(questions[q], opts)[0] for q, opts in zip(qids, opts_l)]
    # ask_branch's split
    L, m = 0, min(len(x) for x in ids_l)
    while L < m and all(x[L] == ids_l[0][L] for x in ids_l):
        L += 1
    L = max(1, min([L] + [sp[0][0] - 1 for sp in spans_l if sp[0]]))
    qrows = [BR.rows_from_ids(row, L, sp[0], sp[1], k=len(opts)) for row, sp, opts in zip(ids_l, spans_l, opts_l)]
    usage = int(L + sum(len(r) for qr in qrows for r in qr))

    def positions(ids):
        enc = {"input_ids": torch.tensor(ids)[None], "image_grid_thw": pix["image_grid_thw"],
               "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return T.MSO.rope_positions(rope, enc, None, False)[:, 0, :]

    out_q = []
    for qid, keys, opts, ids, (opens, closes), prompt, rows in zip(qids, keys_l, opts_l, ids_l, spans_l, prompts, qrows):
        pos = positions(ids)
        # branch semantics: prefix positions from the prefix alone, then text positions from max + 1
        ppos = positions(ids[:L])
        assert torch.equal(ppos, pos[:, :L]), name
        p0 = int(ppos.max()) + 1
        row_readouts = []
        for r in rows:
            o, c, inner = BR._marks(r, obj.open_ids, obj.close_ids)
            row_readouts.append({"tokens": r, "zq": max(0, o - 1), "u": c,
                                 "targets": [[t - 1, r[t]] for t in inner],
                                 "first": r[inner[0]] if inner else None})
            # the row's positions in a full sequence equal p0 + offset
            assert torch.equal(positions(list(ids[:L]) + r)[:, L:], torch.arange(p0, p0 + len(r))[None].expand(3, -1))
        assert prompt.count("<|image_pad|>") == 1        # the processor expands it to the grid's token count
        out_q.append({"id": qid, "type": questions[qid]["type"], "keys": keys,
                      "option_text": [T.render_option(o) for o in opts],
                      "prompt": prompt, "token_ids": collapse(ids),
                      "opens": opens, "closes": closes, "positions_sha256": sha256_i64(pos.tolist()),
                      "rows": row_readouts})
    request = {"model": "tinnel123/OmniJev", "state": {"images": ["<solid PNG of image_size>"]}, "questions": questions}
    pv = pix["pixel_values"].float().numpy()      # for the vision step's pixel check
    return {"name": name, "request": request, "image_size": list(size), "image_colour": list(COLOUR),
            "image_grid_thw": pix["image_grid_thw"][0].tolist(), "prefix_length": L, "input_tokens": usage,
            "pixel_values_sha256": hashlib.sha256(pv.tobytes()).hexdigest(), "pixel_values_shape": list(pv.shape),
            "questions": out_q}


def head_cases(checkpoint):
    from mso.head import OptionScorer, OrdinalScoreHead
    head = OptionScorer(2560, d_hidden=1024, norm="softmax")
    head.load_state_dict(torch.load(os.path.join(checkpoint, "head.pt"), map_location="cpu"), strict=True)
    ordh = OrdinalScoreHead(2560, d_hidden=512)
    ordh.load_state_dict(torch.load(os.path.join(checkpoint, "ord.pt"), map_location="cpu"), strict=True)
    head, ordh = head.float().eval(), ordh.float().eval()
    cases = []
    for seed, (name, type_id, k) in enumerate((("noul", 0, 1), ("choice2", 1, 2), ("choice5", 1, 5), ("score4", 2, 4))):
        u = sequence(3 * seed, k * 2560).reshape(k, 2560)
        zq = sequence(3 * seed + 1, 2560)
        feats = torch.zeros(k, 6) if type_id == 0 else sequence(3 * seed + 2, k * 6).reshape(k, 6)
        with torch.no_grad():
            logits = head.logits(u, zq, type_id, feats)
            mu = head(u, zq, type_id, feats)
            ordinal = ordh(zq, u)
        cases.append({"name": name, "seeds": [3 * seed, 3 * seed + 1, 3 * seed + 2], "type_id": type_id, "options": k,
                      "logits": logits.tolist(), "mu": mu.tolist(), "ordinal": ordinal.tolist()})
    return cases


def feature_cases():
    """branch_forward's six LM features, from a stub backbone and a small lm_head."""
    import mso.branch as BR
    g = torch.Generator().manual_seed(85)
    V, D = 300, 16
    lm_head = torch.nn.Linear(D, V, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(V, D, generator=g))
    open_ids, close_ids = [298], [299]
    head = [5, 6, 7]
    rows = [head + [298, 11, 12, 13, 299], head + [298, 14, 299], head + [298, 299],
            [8, 9] + [298, 20, 21, 22, 23, 24, 25, 299], [8, 9] + [298, 299]]
    groups = [[0, 1, 2], [3], [4]]
    hidden = {i: torch.randn(len(r), D, generator=g) * 3 for i, r in enumerate(rows)}

    class Cache:
        layers = []

    class Backbone:
        def __call__(self, input_ids, **kw):
            if "past_key_values" not in kw:
                return types.SimpleNamespace(past_key_values=Cache())
            n = input_ids.shape[1]
            h = torch.zeros(input_ids.shape[0], n, D)
            for b in range(input_ids.shape[0]):
                key = next(i for i, r in enumerate(rows) if input_ids[b, :len(r)].tolist() == r)
                h[b, :len(rows[key])] = hidden[key]
            return types.SimpleNamespace(hidden_states=[h])

    penc = {"input_ids": torch.tensor([[1, 2, 3]])}
    with torch.no_grad():
        out = BR.branch_forward(Backbone(), penc, rows, torch.device("cpu"), open_ids, close_ids, lm_head=lm_head,
                                groups=groups, chunk=2, pad_id=0, mrope_axes=3)
        cases = []
        for i, r in enumerate(rows):
            o, c, inner = BR._marks(r, open_ids, close_ids)
            lp = torch.log_softmax(lm_head(hidden[i]).float(), dim=-1)
            picked = [float(lp[t - 1, r[t]]) for t in inner]
            first = float(lp[o - 1, r[inner[0]]]) if inner else None
            cases.append({"row": i, "picked": picked, "first": first, "feats": out["feats"][i].tolist()})
    return {"groups": groups, "rows": cases}


def finish_cases(reference, checkpoint):
    from mso.infer import MSO1
    meta = json.load(open(os.path.join(checkpoint, "head_meta.json")))
    stub = MSO1.__new__(MSO1)
    stub.temp = float(meta.get("temperature", 1.0))
    stub.temps = {k: float(meta["temperatures"][k]) for k in ("noul", "choice", "score")}
    stub.biases = {"noul": float(meta["biases"]["noul"])}
    cases = []
    inputs = [
        ("noul", ["yes"], [0.5]), ("noul", ["yes"], [0.4867]), ("noul", ["yes"], [1e-12]), ("noul", ["yes"], [0.999999]),
        ("choice", ["a", "b", "c"], [0.2, 0.3, 0.1]), ("choice", ["a", "b"], [0.45, 0.45]),
        ("choice", ["a", "b", "c", "d"], [0.0001, 0.9, 0.05, 0.0]), ("choice", ["only"], [0.7]),
        ("score", ["low", "mid", "high"], [0.2, 0.5, 0.3]), ("score", ["0", "1"], [1e-8, 1.0]),
        ("score", ["a", "b", "c", "d"], [0.25, 0.25, 0.25, 0.25]),
    ]
    g = torch.Generator().manual_seed(61)
    for k in (5, 12, 32):
        logits = torch.randn(k + 1, generator=g) * 2
        inputs.append(("choice", [f"o{i}" for i in range(k)], torch.softmax(logits, 0)[:k].tolist()))
        levels = torch.rand(k, generator=g) + 1e-3
        inputs.append(("score", [f"l{i}" for i in range(k)], (levels / levels.sum()).tolist()))
    inputs.append(("choice", ["a", "b", "c", "d", "e"], [0.3, 0.3, 0.2, 0.15, 0.1]))   # sums above one: invalid
    for qtype, keys, mu in inputs:
        q = {"type": qtype, "instructions": "x"}
        out = MSO1._finish(stub, q, keys, [{"text": k} for k in keys], torch.tensor(mu), 0.0)
        cases.append({"type": qtype, "keys": keys, "mu": torch.tensor(mu).tolist(), "answer": out})
    return {"temperatures": stub.temps, "biases": stub.biases, "cases": cases}


def pattern(width, height):
    """The test pattern tests/omnijev/pixels.rs draws too: channel values from x and y."""
    import numpy as np
    x = np.arange(width, dtype=np.int64)[None, :]
    y = np.arange(height, dtype=np.int64)[:, None]
    return np.stack([(x * 7 + y * 3) % 256, (x * y + 13) % 256, ((x ^ y) * 5) % 256], -1).astype(np.uint8)


def pixel_cases(obj):
    """Images as the reference reads them, `Image.open(path).convert("RGB")`, and the
    processor's pixel values for them. Pattern PNGs are redrawn by the test; files whose
    encoding matters (palette, 16-bit, JPEG) are stored."""
    import base64
    import numpy as np
    from PIL import Image

    def encode(image, fmt, **kw):
        buf = io.BytesIO()
        image.save(buf, format=fmt, **kw)
        return buf.getvalue()

    rgba = np.concatenate([pattern(300, 200), ((np.arange(300)[None, :] + np.arange(200)[:, None]) % 256)
                           .astype(np.uint8)[..., None]], -1)
    gray16 = ((np.arange(160)[None, :] * 400 + np.arange(100)[:, None] * 50) % 65536).astype(np.uint16)
    files = [
        ("rgb_upscale", "pattern", encode(Image.fromarray(pattern(64, 48)), "PNG")),
        ("rgb_odd", "pattern", encode(Image.fromarray(pattern(333, 217)), "PNG")),
        ("rgb_downscale", "pattern", encode(Image.fromarray(pattern(1920, 1080)), "PNG")),
        ("rgb_wide", "pattern", encode(Image.fromarray(pattern(2000, 20)), "PNG")),
        ("rgba", "pattern_rgba", encode(Image.fromarray(rgba, "RGBA"), "PNG")),
        ("gray", "pattern_gray", encode(Image.fromarray(pattern(200, 150)[..., 0], "L"), "PNG")),
        ("palette", "stored", encode(Image.fromarray(pattern(180, 120)).quantize(64), "PNG")),
        ("gray16", "stored", encode(Image.fromarray(gray16, "I;16"), "PNG")),
        ("jpeg", "stored", encode(Image.fromarray(pattern(96, 64)), "JPEG", quality=90)),
    ]
    cases = []
    for name, kind, raw in files:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        rgb = np.asarray(im, dtype=np.uint8)
        enc = obj.proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[im], return_tensors="pt")
        pixels = enc["pixel_values"].to(torch.float32).contiguous()
        case = {"name": name, "kind": kind, "format": "jpeg" if raw[:2] == b"\xff\xd8" else "png",
                "size": [im.width, im.height], "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "image_grid_thw": enc["image_grid_thw"][0].tolist(), "pixel_values_shape": list(pixels.shape),
                "pixel_values_sha256": hashlib.sha256(pixels.numpy().tobytes()).hexdigest()}
        if kind == "stored":
            case["bytes"] = base64.b64encode(raw).decode()
        if case["format"] == "jpeg":
            case["rgb"] = base64.b64encode(rgb.tobytes()).decode()
        cases.append(case)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import transformers
    obj = make_reference(a.reference, a.checkpoint)
    rope = rope_owner(a.base)
    cases = []
    for name, size, questions in CASES:
        qs = speed_bench_questions(a.reference) if questions is None else copy.deepcopy(questions)
        cases.append(processing_case(obj, rope, name, size, qs))
    provenance = {"python": sys.version.split()[0], "reference_commit": REFERENCE_COMMIT, "checkpoint_revision": CHECKPOINT_REVISION,
                  "base_revision": BASE_REVISION, "max_pixels": MAX_PIXELS, "torch": torch.__version__,
                  "transformers": transformers.__version__,
                  "command": "CUDA_VISIBLE_DEVICES= python tests/omnijev/generate_fixtures.py"}
    os.makedirs(a.out, exist_ok=True)
    for fname, payload in (("processing.json", {"provenance": provenance, "cases": cases}),
                           ("pixels.json", {"provenance": provenance, "cases": pixel_cases(obj)}),
                           ("heads.json", {"provenance": provenance, "cases": head_cases(a.checkpoint)}),
                           ("features.json", {"provenance": provenance, **feature_cases()}),
                           ("finish.json", {"provenance": provenance, **finish_cases(a.reference, a.checkpoint)})):
        with open(os.path.join(a.out, fname), "w") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        print("wrote", fname, os.path.getsize(os.path.join(a.out, fname)), "bytes")


if __name__ == "__main__":
    main()
