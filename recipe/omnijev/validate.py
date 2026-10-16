#!/usr/bin/env python3
"""Compare the native worker with the pinned reference on the same images and questions.

    # the reference, in the reference environment (README.md) on one GPU
    python recipe/omnijev/validate.py reference --reference OmniJev --checkpoint omnijev-v1.1 \
        --base qwen3.5-4b --images IMAGES --out reference.json
    # the native worker, started as in native.md, with the questions and images recorded above
    python recipe/omnijev/validate.py native --url http://127.0.0.1:8000 --images IMAGES \
        --against reference.json --out native.json
    python recipe/omnijev/validate.py compare reference.json native.json

The questions are bench/speed_bench.py's twelve, or a JSON map of id to question given with
--questions. The reference runs MSO1 as released (its branch path); with --merge it first
merges the LoRA into the BF16 weights as export.py does, which isolates the merge's effect;
--fp32 loads the model in float32 with TF32 off, the baseline for BF16 rounding; and
MSO_BRANCH_CHUNK=1 runs its rows one at a time. The worker gets each image as a data
URL; both record the SHA-256 of every image file.
"""
import argparse
import base64
import hashlib
import json
import os
import statistics
import sys
import urllib.error
import urllib.request


def images(directory):
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith((".png", ".jpg", ".jpeg")))
    if not names:
        sys.exit(f"no PNG or JPEG images in {directory}")
    return [(n, os.path.join(directory, n)) for n in names]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def reference(a):
    import torch
    import transformers
    sys.path.insert(0, a.reference)
    from mso.infer import MSO1
    if a.questions:
        questions = json.load(open(a.questions))
    else:
        from bench.speed_bench import Q
        questions = dict(Q)
    if a.fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        load = transformers.AutoModelForImageTextToText.from_pretrained
        transformers.AutoModelForImageTextToText.from_pretrained = lambda *p, **kw: load(*p, **{**kw, "dtype": torch.float32})
    model = MSO1(a.checkpoint, a.base)
    if a.merge:
        from mso import v04
        model.model.backbone = model.model.backbone.merge_and_unload(safe_merge=True)
        model.lm = v04.find_lm_head(model.model.backbone) if model.lm_feats else None
    if not model.branch:
        sys.exit("the reference is not on its branch path (MSO_BRANCH=0?), which the worker reproduces")
    files = images(a.images)
    model.system_one({"images": [files[0][1]]}, questions)  # warm-up
    out = {"environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                           "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                           "branch": model.branch, "merged": a.merge, "fp32": a.fp32,
                           "branch_chunk": os.environ.get("MSO_BRANCH_CHUNK", "64")},
           "questions": questions, "images": {}}
    for name, path in files:
        answers = model.system_one({"images": [path]}, questions)
        out["images"][name] = {"sha256": sha256(path), "input_tokens": model.last_input_tokens, "answers": answers}
        print(name, "done", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


def native(a):
    against = json.load(open(a.against))
    out = {"url": a.url, "images": {}}
    for name, path in images(a.images):
        expected = against["images"].get(name)
        if expected is None or expected["sha256"] != sha256(path):
            sys.exit(f"{name} differs from the image in {a.against}")
        mime = "image/png" if open(path, "rb").read(4) == b"\x89PNG" else "image/jpeg"
        url = f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
        body = json.dumps({"model": "tinnel123/OmniJev", "state": {"images": [url]},
                           "questions": against["questions"]}).encode()
        request = urllib.request.Request(a.url.rstrip("/") + "/v1/systemone", body,
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as r:
                response = json.load(r)
        except urllib.error.HTTPError as e:
            out["images"][name] = {"sha256": expected["sha256"], "status": e.code, "detail": e.read().decode()}
            print(name, "failed with", e.code, flush=True)
            continue
        out["images"][name] = {"sha256": expected["sha256"], "input_tokens": response["usage"]["input_tokens"],
                               "answers": response["answers"]}
        print(name, "done", flush=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


def values(answer):
    """Every probability-like number of an answer, by name."""
    out = {}
    for key in ("noul", "abstain", "confidence"):
        if key in answer:
            out[key] = answer[key]
    for key, p in answer.get("probabilities", {}).items():
        out["p:" + key] = p
    return out


def compare(a):
    ref, nat = json.load(open(a.reference_json)), json.load(open(a.native_json))
    diffs, flips, problems, tokens, total = [], [], [], 0, 0
    for name, r in ref["images"].items():
        n = nat["images"].get(name)
        if n is None or "answers" not in n:
            problems.append(f"{name}: {'missing' if n is None else 'status %s %s' % (n['status'], n['detail'])}")
            continue
        tokens += r["input_tokens"] != n["input_tokens"]
        for qid, ra in r["answers"].items():
            na = n["answers"].get(qid)
            rv, nv = values(ra), values(na or {})
            if na is None or rv.keys() != nv.keys():
                problems.append(f"{name} {qid}: fields {sorted(rv)} vs {sorted(nv)}")
                continue
            total += 1
            worst = max(abs(rv[k] - nv[k]) for k in rv)
            diffs.append(worst)
            decision = next((k for k in ("choice", "score") if k in ra), None)
            if decision and ra[decision] != na[decision]:
                p = sorted(ra["probabilities"].values(), reverse=True)
                margin = p[0] - p[1] if len(p) > 1 else 1.0
                flips.append((name, qid, ra[decision], na[decision], margin))
            if decision is None and (ra["noul"] >= 0.5) != (na["noul"] >= 0.5):
                flips.append((name, qid, ra["noul"], na["noul"], abs(ra["noul"] - 0.5)))
    for p in problems:
        print("not compared:", p)
    if not diffs:
        sys.exit("nothing to compare")
    diffs.sort()
    q = lambda f: diffs[min(len(diffs) - 1, int(f * len(diffs)))]
    print(f"{total} answers on {len(ref['images'])} images; input tokens differ on {tokens} images")
    print(f"largest difference per answer: median {statistics.median(diffs):.4f}, p90 {q(0.9):.4f}, "
          f"p99 {q(0.99):.4f}, max {diffs[-1]:.4f}; within 0.01: {sum(d <= 0.01 for d in diffs)}")
    print(f"decision changes: {len(flips)}")
    for f in flips:
        print("  {} {}: reference {} native {} (reference margin {:.4f})".format(*f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--reference", required=True, help="OmniJev checkout @ 14dbec4")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--images", required=True)
    r.add_argument("--questions")
    r.add_argument("--merge", action="store_true", help="merge the LoRA into the BF16 weights first")
    r.add_argument("--fp32", action="store_true", help="load the model in float32, TF32 off")
    r.add_argument("--out", required=True)
    n = sub.add_parser("native")
    n.add_argument("--url", required=True)
    n.add_argument("--images", required=True)
    n.add_argument("--against", required=True, help="the reference output, for its questions and image hashes")
    n.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("reference_json")
    c.add_argument("native_json")
    a = ap.parse_args()
    {"reference": reference, "native": native, "compare": compare}[a.command](a)


if __name__ == "__main__":
    main()
