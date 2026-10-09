"""Export the pinned BF16 language model with the multimodal LoRA merged.

Vision execution keeps its original BF16 base and FP32 adapters separately.
This one-time export requires the pinned Transformers/PEFT reference environment.
"""

import argparse
import hashlib
import json
from pathlib import Path

from models.cua_s1.multimodal.model import (
    ADAPTER_REVISION,
    BASE_REVISION,
    MultimodalEngine,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    engine = MultimodalEngine(args.base, args.adapter)
    model = engine.model.merge_and_unload()
    model.model.language_model.save_pretrained(args.out, max_shard_size="5GB")
    model.config.to_json_file(args.out / "config.json")
    engine.tokenizer.save_pretrained(args.out)
    (args.out / "cua_s1_language_export.json").write_text(
        json.dumps(
            {
                "format": "cua-s1-multimodal-language-merged/1",
                "base_revision": BASE_REVISION,
                "adapter_revision": ADAPTER_REVISION,
                "vision": "separate unmerged base and adapter",
                "files": {
                    p.name: {
                        "size": p.stat().st_size,
                        "sha256": hashlib.file_digest(
                            p.open("rb"), "sha256"
                        ).hexdigest(),
                    }
                    for p in args.out.iterdir()
                    if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
