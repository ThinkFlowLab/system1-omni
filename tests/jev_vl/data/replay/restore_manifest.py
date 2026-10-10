#!/usr/bin/env python3
"""Restore the frozen historical manifest, verifying its original SHA-256."""

import argparse
import base64
import hashlib
from pathlib import Path

EXPECTED_SHA256 = "f72d1beaaaf53933d0a6edda26b635f46931990d8cdb56ca5c6a7ca94d2eb0ee"
MARKER = b"__JEV_VL_FROZEN_IMAGE_DATA_URI__"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    template = (source / "manifest-template.jsonl").read_bytes()
    image = (source / "image.png").read_bytes()
    uri = b"data:image/png;base64," + base64.b64encode(image)
    restored = template.replace(MARKER, uri)
    digest = hashlib.sha256(restored).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"manifest checksum mismatch: {digest}")
    if args.out.exists() and args.out.read_bytes() != restored:
        raise FileExistsError(f"refusing to replace a different manifest: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(restored)
    print(f"48 requests, {len(restored)} bytes, sha256={digest}")


if __name__ == "__main__":
    main()
