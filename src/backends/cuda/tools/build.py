"""Compile an exported bundle. Preserve separate TileLang/PyTorch arithmetic flags."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    root = Path(__file__).resolve().parents[1]
    nvcc = str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc")
    commands = []
    objects = []
    sources = {}

    for source, fast_math in [
        (args.bundle / "generated.cu", True),
        (root / "kernels/runtime.cu", False),
        (root / "kernels/model_ops.cu", False),
    ]:
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        obj = args.bundle / (source.stem + ".o")
        objects.append(str(obj))
        flags = [
            flag
            for flag in manifest["nvcc_flags"]
            if fast_math or flag != "--use_fast_math"
        ]
        command = [
            nvcc,
            *flags,
            "--expt-relaxed-constexpr",
            "-c",
            "-Xcompiler=-fPIC",
            "-O3",
            *[
                arg
                for directory in manifest["include_dirs"]
                for arg in ["-I", directory]
            ],
            str(source),
            "-o",
            str(obj),
        ]
        commands.append(command)
        subprocess.run(command, check=True)

    library = args.bundle / "liblaya_cuda.so"
    command = [nvcc, "-shared", *objects, "-lcublas", "-lcuda", "-o", str(library)]
    commands.append(command)
    subprocess.run(command, check=True)
    (args.bundle / "build-command.json").write_text(json.dumps(commands, indent=2))

    build_manifest = {
        "abi": 1,
        "arch": "sm_90a",
        "nvcc": subprocess.check_output([nvcc, "--version"], text=True),
        "sources": sources,
        "commands": commands,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    (args.bundle / "build-manifest.json").write_text(
        json.dumps(build_manifest, indent=2)
    )


if __name__ == "__main__":
    main()
