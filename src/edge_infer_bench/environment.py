"""Environment inventory; imports/probes are opt-in to avoid polluting benchmark RSS."""

import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil


def environment_info(*, probe: bool = False) -> dict[str, Any]:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    versions: dict[str, Any] = {}
    for package in (
        "torch",
        "torchvision",
        "onnx",
        "onnxscript",
        "onnxruntime",
        "openvino",
        "numpy",
        "psutil",
        "edge-infer-bench",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    process = psutil.Process()
    governors = sorted(
        {
            p.read_text().strip()
            for p in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")
        }
    )
    frequencies = []
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_cur_freq"):
        try:
            frequencies.append(int(path.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    thermal = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            thermal.append({"path": str(path), "celsius": int(path.read_text().strip()) / 1000.0})
        except (OSError, ValueError):
            continue
    git_commit = None
    git_dirty = None
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    result: dict[str, Any] = {
        "os": platform.platform(),
        "distribution": platform.freedesktop_os_release()
        if Path("/etc/os-release").exists()
        else {},
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "python": sys.version.split()[0],
        "versions": versions,
        "cpu_affinity": process.cpu_affinity() if hasattr(process, "cpu_affinity") else None,
        "governors": governors,
        "system_memory_mb": psutil.virtual_memory().total / 1e6,
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "KMP_AFFINITY",
            )
        },
        "kernel": platform.release(),
        "cpu_frequency_mhz": {
            "current_min": min(frequencies) if frequencies else None,
            "current_max": max(frequencies) if frequencies else None,
            "sample_count": len(frequencies),
        },
        "thermal": thermal or None,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if probe:
        backends = {}
        for name, module in (
            ("pytorch", "torch"),
            ("onnxruntime", "onnxruntime"),
            ("openvino", "openvino"),
        ):
            try:
                loaded = importlib.import_module(module)
                details: dict[str, Any] = {"status": "importable", "version": loaded.__version__}
                if name == "onnxruntime":
                    details["providers"] = loaded.get_available_providers()
                    if "CPUExecutionProvider" not in details["providers"]:
                        details["status"] = "unavailable"
                elif name == "openvino":
                    details["devices"] = loaded.Core().available_devices
                    details["cpu_execution"] = "unverified: conversion/inference must succeed"
                    if "CPU" not in details["devices"]:
                        details["status"] = "unavailable"
                backends[name] = details
            except Exception as exc:
                backends[name] = {"status": "unavailable", "reason": str(exc)}
        result["backends"] = backends
    return result
