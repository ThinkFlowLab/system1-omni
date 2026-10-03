"""Run header: everything needed to tell whether two result files are comparable.

Standard library plus whatever the benchmark already imported; every probe degrades to None
instead of failing the run.
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(*cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _checkpoint_revision(repo_id, ref="main"):
    """The commit the cached ref points at. Read directly: laya fetches only the files it needs, so
    the snapshot is partial and snapshot_download(local_files_only=True) refuses it."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        ref_file = (
            Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "refs" / ref
        )
        return ref_file.read_text().strip()
    except (ImportError, OSError):
        return None


def _power_source():
    out = _run("pmset", "-g", "batt")
    if not out:
        return None
    first = out.splitlines()[0]
    return first.split("'")[1] if "'" in first else first


def _gpu_cores():
    out = _run("system_profiler", "SPDisplaysDataType")
    for line in (out or "").splitlines():
        if "Total Number of Cores" in line:
            return int(line.split(":")[1])
    return None


def noise_problems(max_load):
    """Reasons this machine is not fit for a measured run, empty when it is."""
    problems = []
    power = _power_source()
    if power and power != "AC Power":
        problems.append(f"on {power}")
    load = os.getloadavg()[0]
    if load > max_load:
        top = _run("ps", "-Ao", "pcpu=,comm=", "-r") or ""
        busiest = "; ".join(
            " ".join(line.split()[:1] + [line.split("/")[-1]])
            for line in top.splitlines()[:3]
        )
        problems.append(f"1-min load {load:.1f} > {max_load} (busiest: {busiest})")
    return problems


def refuse_if_noisy(max_load, measured):
    """Exits when a measured run would be noisy; otherwise warns about each problem and returns them."""
    problems = noise_problems(max_load)
    if problems and measured:
        sys.exit("refusing a measured run: " + "; ".join(problems))
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    return problems


def read_workloads(path):
    """The fixed inputs by id, in file order."""
    workloads = {}
    with open(path) as f:
        for w in map(json.loads, filter(str.strip, f)):
            if w["id"] in workloads:
                raise ValueError(f"{path}: workload id {w['id']} appears twice")
            workloads[w["id"]] = w
    return workloads


def header(checkpoint, **extra):
    status = _run(
        "git",
        "-C",
        str(REPO),
        "status",
        "--porcelain",
        "--",
        ".",
        ":!benchmarks/laya_mps/results",
    )
    return {
        "type": "env",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "omni_sha": _run("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "omni_dirty": bool(status),
        "checkpoint": checkpoint,
        "checkpoint_revision": _checkpoint_revision(checkpoint),
        "laya": _version("laya"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "python": platform.python_version(),
        "os": f"macOS {platform.mac_ver()[0]}"
        if sys.platform == "darwin"
        else platform.platform(),
        "chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_perf_cores": _run("sysctl", "-n", "hw.perflevel0.physicalcpu"),
        "cpu_eff_cores": _run("sysctl", "-n", "hw.perflevel1.physicalcpu"),
        "gpu_cores": _gpu_cores(),
        "mem_gb": round(int(_run("sysctl", "-n", "hw.memsize") or 0) / 2**30),
        "power": _power_source(),
        "loadavg_1m": round(os.getloadavg()[0], 2),
        "argv": sys.argv,
        **extra,
    }


RUSAGE_INFO_V4 = 4  # <sys/resource.h>


def footprint_mb(pid=None):
    """Physical footprint of a process, now and its lifetime peak (MB), from proc_pid_rusage.

    This is Activity Monitor's "Memory" column. On Apple silicon it includes Metal allocations, so it
    covers MPS tensors that RSS misses, and it reads the same way for this process and a worker's pid.
    """
    import ctypes

    class RusageInfoV4(ctypes.Structure):  # <sys/resource.h>, rusage_info_v4
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64)
            for name in [
                "user_time",
                "system_time",
                "pkg_idle_wkups",
                "interrupt_wkups",
                "pageins",
                "wired_size",
                "resident_size",
                "phys_footprint",
                "proc_start_abstime",
                "proc_exit_abstime",
                "child_user_time",
                "child_system_time",
                "child_pkg_idle_wkups",
                "child_interrupt_wkups",
                "child_pageins",
                "child_elapsed_abstime",
                "diskio_bytesread",
                "diskio_byteswritten",
                "cpu_time_qos_default",
                "cpu_time_qos_maintenance",
                "cpu_time_qos_background",
                "cpu_time_qos_utility",
                "cpu_time_qos_legacy",
                "cpu_time_qos_user_initiated",
                "cpu_time_qos_user_interactive",
                "billed_system_time",
                "serviced_system_time",
                "logical_writes",
                "lifetime_max_phys_footprint",
                "instructions",
                "cycles",
                "billed_energy",
                "serviced_energy",
                "interval_max_phys_footprint",
                "runnable_time",
            ]
        ]

    if sys.platform != "darwin":
        return {}
    info = RusageInfoV4()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    if (
        libc.proc_pid_rusage(pid or os.getpid(), RUSAGE_INFO_V4, ctypes.byref(info))
        != 0
    ):
        return {}
    return {
        "footprint_mb": round(info.phys_footprint / 2**20),
        "footprint_peak_mb": round(info.lifetime_max_phys_footprint / 2**20),
    }
