"""Export Qwen3.5-4B with the Cua-S1 `text` adapter merged into the bfloat16 weights,
for the native worker (recipe/cua_s1/native.md). Run it in the reference worker's
environment (recipe/cua_s1/text.md):

    PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_text_merged.py \
        --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text \
        --out weights/cua-s1-4b-0.2-text-merged
"""

import argparse
import json
from pathlib import Path

from models.cua_s1.text.model import TextModel

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--base", required=True)
parser.add_argument("--adapter", required=True)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()

loaded = TextModel(args.base, args.adapter, "cuda", "bfloat16")
loaded.model.merge_and_unload().save_pretrained(args.out, max_shard_size="5GB")
# Transformers writes the pre-tokenizer rule it uses into tokenizer.json; the native
# worker tokenizes with that file.
loaded.tokenizer.save_pretrained(args.out)
# The native worker refuses a directory without this marker, such as the base model.
(args.out / "cua_s1_export.json").write_text(
    json.dumps({"format": "cua-s1-text-merged/1"})
)
