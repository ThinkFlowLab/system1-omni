#!/usr/bin/env python3
"""Download and SHA-256 verify the immutable Decider-2B v11 artifacts."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = "533964dae8be954c5b5e19fa4948e48408094c1e"
FILES = {
    "config.json": (1790, "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52"),
    "decider_config.json": (1240, "6e4891f2754a1c18a10f8dadb0c04e439e7f79fab0333d56641491bd4a05e722"),
    "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    "tokenizer_config.json": (1127, "171ecbe7ddae98d11840698f7df2b8d5b4722139db0f0620d3bbf429bd656250"),
    "model.safetensors": (3763692048, "acaef2228b134dcdc20cad4ee79219482c927ec819aa3687b9b8a575c338817f"),
}


def verify(path, expected):
    size, digest = expected
    if path.stat().st_size != size:
        raise ValueError(f"{path}: size mismatch")
    actual = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            actual.update(block)
    if actual.hexdigest() != digest:
        raise ValueError(f"{path}: checksum mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = args.directory / name
        if destination.exists():
            verify(destination, expected)
        else:
            fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=args.directory)
            temporary = Path(temporary)
            try:
                url = f"{args.endpoint.rstrip('/')}/Mapika/decider-2b/resolve/{REVISION}/{name}"
                with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=120) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
                verify(temporary, expected)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"verified {name}", flush=True)


if __name__ == "__main__":
    main()
