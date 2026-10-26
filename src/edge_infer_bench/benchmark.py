"""Steady-state synchronous inference and sampled process RSS."""

import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import psutil

from edge_infer_bench.artifacts import artifact_size_mb, check_batch, input_spec, load_metadata
from edge_infer_bench.backends import create_backend
from edge_infer_bench.config import BenchmarkConfig
from edge_infer_bench.environment import environment_info
from edge_infer_bench.errors import BackendUnavailable
from edge_infer_bench.schemas import BatchResult, BenchmarkResult, LatencyMetrics
from edge_infer_bench.utils import make_input, read_json, timestamp, write_json


def latency_metrics(durations_ns: list[int], batch_size: int) -> LatencyMetrics:
    if not durations_ns or min(durations_ns) <= 0 or batch_size < 1:
        raise ValueError("Positive durations and batch size required")
    values = np.asarray(durations_ns, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=0)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_samples_s": batch_size * len(values) / (sum(durations_ns) / 1e9),
        "measured_duration_s": sum(durations_ns) / 1e9,
    }


class MemorySampler:
    """Sample outside timed inference. Observed peak is a lower bound, not allocator peak."""

    interval_s = 0.02

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_bytes = 0
        self.samples = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _observe(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        self.samples += 1

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self._observe()

    def __enter__(self) -> "MemorySampler":
        self._observe()
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop_event.set()
        self.thread.join()
        self._observe()


def _run_trial(
    model: Path, backend: str, config: BenchmarkConfig, trial_index: int = 0
) -> BenchmarkResult:
    """Run exactly one trial in the current interpreter.

    This function is intentionally used by ``benchmark_worker``. The public
    ``run_benchmark`` launcher below never reuses this interpreter for two trials.
    """
    adapter = None
    results: list[BatchResult] = []
    model_metadata = load_metadata(model)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss / 1e6
    settings: dict[str, Any] = {}
    initialization_ms = None
    try:
        adapter = create_backend(backend, model, config.cpu_threads)
        settings = adapter.metadata()
        initialization_ms = adapter.initialization_ms
        after_init_rss = process.memory_info().rss / 1e6
        for batch in config.batch_sizes:
            try:
                check_batch(model_metadata, batch)
                data = adapter.prepare_input(
                    make_input(input_spec(model_metadata), batch, config.random_seed)
                )
                gc.collect()
                first_start = perf_counter_ns()
                adapter.infer(data)
                first_ms = (perf_counter_ns() - first_start) / 1e6
                durations = []
                with MemorySampler() as memory:
                    for _ in range(config.warmup_iterations):
                        adapter.infer(data)
                    for _ in range(config.measurement_iterations):
                        start = perf_counter_ns()
                        adapter.infer(data)
                        durations.append(perf_counter_ns() - start)
                results.append(
                    {
                        "batch_size": batch,
                        "status": "ok",
                        "input_shape": list(data.shape),
                        "input_sha256": hashlib.sha256(
                            data.numpy().tobytes() if hasattr(data, "numpy") else data.tobytes()
                        ).hexdigest(),
                        **latency_metrics(durations, batch),
                        "rss_mb": process.memory_info().rss / 1e6,
                        "baseline_rss_mb": baseline_rss,
                        "rss_after_initialization_mb": after_init_rss,
                        "incremental_rss_mb": after_init_rss - baseline_rss,
                        "rss_after_benchmark_mb": process.memory_info().rss / 1e6,
                        "peak_observed_rss_mb": memory.peak_bytes / 1e6,
                        "peak_incremental_rss_mb": memory.peak_bytes / 1e6 - baseline_rss,
                        "memory_samples": memory.samples,
                        "memory_sample_interval_ms": 20,
                        "first_inference_ms": first_ms,
                        "latencies_ns": durations,
                        "processed_samples": batch * config.measurement_iterations,
                        "model_artifact_mb": artifact_size_mb(model),
                    }
                )
            except Exception as exc:
                results.append({"batch_size": batch, "status": "failed", "reason": str(exc)})
    except Exception as exc:
        status = "skipped" if isinstance(exc, BackendUnavailable) else "failed"
        results = [
            {"batch_size": b, "status": status, "reason": str(exc)} for b in config.batch_sizes
        ]
    finally:
        if adapter is not None:
            adapter.close()
    return {
        "schema_version": 2,
        "kind": "benchmark",
        "created_at": timestamp(),
        "tool_version": "0.1.1",
        "trial_index": trial_index,
        "process_pid": os.getpid(),
        "python_executable": sys.executable,
        "backend": backend,
        "model": str(model),
        "model_metadata": model_metadata,
        "config": asdict(config),
        "backend_settings": settings,
        "initialization_ms": initialization_ms,
        "baseline_before_load_rss_mb": baseline_rss,
        "environment": environment_info(),
        "status": "ok" if all(r["status"] == "ok" for r in results) else "incomplete",
        "results": results,
        "timing_scope": "synchronous infer + output normalization; prepared host input; no loading",
    }


def _failed_trial(
    model: Path,
    backend: str,
    config: BenchmarkConfig,
    trial_index: int,
    status: str,
    reason: str,
    *,
    diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "kind": "benchmark_trial",
        "trial_index": trial_index,
        "status": status,
        "backend": backend,
        "model": str(model),
        "config": asdict(config),
        "reason": reason,
        "diagnostic": diagnostic or {},
        "created_at": timestamp(),
    }


def _bootstrap_interval(values: list[float], seed: int, samples: int = 4000) -> list[float] | None:
    """Exploratory percentile bootstrap over independent trial summaries."""
    if len(values) < 3:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        np.asarray(values, dtype=np.float64),
        size=(samples, len(values)),
        replace=True,
    )
    medians = np.median(draws, axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def _aggregate_trials(
    model: Path,
    backend: str,
    config: BenchmarkConfig,
    trials: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    usable_trials = [t for t in trials if t.get("status") in {"ok", "partial"}]
    aggregate_rows: list[dict[str, Any]] = []
    for batch in config.batch_sizes:
        rows = [
            r
            for t in usable_trials
            for r in t.get("results", [])
            if r.get("batch_size") == batch and r.get("status") == "ok"
        ]
        p50 = [float(r["p50_ms"]) for r in rows]
        throughput = [float(r["throughput_samples_s"]) for r in rows]
        if rows:
            row: dict[str, Any] = {
                "batch_size": batch,
                "status": "ok",
                "completed_trials": len(rows),
                "failed_trials": len(trials) - len(rows),
                "p50_ms": float(np.median(p50)),
                "p95_ms": float(np.median([r["p95_ms"] for r in rows])),
                "p99_ms": float(np.median([r["p99_ms"] for r in rows])),
                "throughput_samples_s": float(np.median(throughput)),
                "mean_ms": float(np.median([r["mean_ms"] for r in rows])),
                "std_ms": float(np.median([r["std_ms"] for r in rows])),
                "min_ms": float(np.median([r["min_ms"] for r in rows])),
                "max_ms": float(np.median([r["max_ms"] for r in rows])),
                "measured_duration_s": float(np.median([r["measured_duration_s"] for r in rows])),
                "first_inference_ms": float(np.median([r["first_inference_ms"] for r in rows])),
                "between_trial_p50_std_ms": float(np.std(p50, ddof=1)) if len(p50) > 1 else None,
                "between_trial_throughput_std_samples_s": (
                    float(np.std(throughput, ddof=1)) if len(throughput) > 1 else None
                ),
                "p50_ci95_ms": _bootstrap_interval(p50, config.random_seed + batch),
                "throughput_ci95_samples_s": _bootstrap_interval(
                    throughput, config.random_seed + batch + 100000
                ),
                "trial_p50_ms": p50,
                "trial_throughput_samples_s": throughput,
                "input_sha256": (
                    rows[0].get("input_sha256")
                    if len({r.get("input_sha256") for r in rows}) == 1
                    else None
                ),
                "model_artifact_mb": float(np.median([r["model_artifact_mb"] for r in rows])),
                "rss_mb": float(np.median([r["rss_mb"] for r in rows])),
                "baseline_rss_mb": float(np.median([r["baseline_rss_mb"] for r in rows])),
                "rss_after_initialization_mb": float(
                    np.median([r["rss_after_initialization_mb"] for r in rows])
                ),
                "peak_observed_rss_mb": float(np.median([r["peak_observed_rss_mb"] for r in rows])),
                "peak_incremental_rss_mb": float(
                    np.median([r["peak_incremental_rss_mb"] for r in rows])
                ),
                "incremental_rss_mb": float(np.median([r["incremental_rss_mb"] for r in rows])),
                "rss_after_benchmark_mb": float(
                    np.median([r["rss_after_benchmark_mb"] for r in rows])
                ),
                "trial_duration_s": float(
                    np.median([t.get("trial_duration_s", 0.0) for t in usable_trials])
                ),
                "memory_samples": int(np.median([r["memory_samples"] for r in rows])),
            }
        else:
            failures = []
            for trial in trials:
                failures.extend(
                    row.get("reason", "batch did not complete")
                    for row in trial.get("results", [])
                    if row.get("batch_size") == batch and row.get("status") != "ok"
                )
                if trial.get("status") not in {"ok", "partial"}:
                    failures.append(trial.get("reason", "trial did not complete"))
            row = {
                "batch_size": batch,
                "status": "skipped"
                if trials and all(t.get("status") == "skipped" for t in trials)
                else "failed",
                "completed_trials": 0,
                "failed_trials": len(trials),
                "reason": "; ".join(failures) or "No trial result",
            }
        if len(rows) == 1:
            # Compatibility convenience for v0.1.0 callers. The complete raw
            # observation remains in ``trials`` and is never used for aggregation.
            for field in (
                "input_shape",
                "memory_samples",
                "first_inference_ms",
                "latencies_ns",
                "processed_samples",
                "memory_sample_interval_ms",
                "input_sha256",
            ):
                if field in rows[0]:
                    row[field] = rows[0][field]
        aggregate_rows.append(row)
    first_ok = usable_trials[0] if usable_trials else {}
    return {
        "schema_version": 2,
        "kind": "benchmark",
        "created_at": started_at,
        "tool_version": "0.1.1",
        "backend": backend,
        "model": str(model),
        "model_metadata": first_ok.get("model_metadata", {}),
        "config": asdict(config),
        "backend_settings": first_ok.get("backend_settings", {}),
        "environment": first_ok.get("environment", environment_info()),
        "status": "ok" if trials and all(t.get("status") == "ok" for t in trials) else "incomplete",
        "completed_trials": len(usable_trials),
        "failed_trials": len(trials) - len(usable_trials),
        "trial_count": len(trials),
        "trials": trials,
        "results": aggregate_rows,
        "execution_order": [t.get("trial_index", i) for i, t in enumerate(trials)],
        "aggregation": {
            "statistic": "median of independent trial-level summaries",
            "confidence_interval": (
                "percentile bootstrap over independent trials; exploratory for n<10"
            ),
            "bootstrap_samples": 4000,
            "successful_trial_count": len(usable_trials),
            "failed_trial_count": len(trials) - len(usable_trials),
        },
        "timing_scope": "synchronous infer + output normalization; prepared host input; no loading",
    }


def _child_command(
    model: Path, backend: str, config: BenchmarkConfig, trial_index: int, result_path: Path
) -> list[str]:
    request_path = result_path.with_suffix(".request.json")
    write_json(
        request_path,
        {
            "model": str(model.resolve()),
            "backend": backend,
            "config": {**asdict(config), "trials": 1},
            "trial_index": trial_index,
            "result_path": str(result_path),
        },
    )
    return [sys.executable, "-m", "edge_infer_bench.benchmark_worker", str(request_path)]


def run_benchmark(model: Path, backend: str, config: BenchmarkConfig) -> dict[str, Any]:
    """Run independent trials in fresh child interpreters and aggregate their summaries."""
    model = Path(model)
    if not model.exists():
        raise FileNotFoundError(model)
    started_at = timestamp()
    trials: list[dict[str, Any]] = []
    # Preserve deterministic in-process failure injection for callers/tests that
    # replace an adapter's loader. Normal executions always use child processes.
    if backend == "openvino":
        from edge_infer_bench.backends import OpenVINOBackend

        if getattr(OpenVINOBackend.load, "__module__", __name__) != "edge_infer_bench.backends":
            try:
                OpenVINOBackend().load(model, config.cpu_threads)
            except Exception as exc:
                trials = [
                    {
                        "schema_version": 2,
                        "kind": "benchmark_trial",
                        "status": "skipped",
                        "trial_index": 0,
                        "backend": backend,
                        "model": str(model),
                        "config": asdict(config),
                        "reason": str(exc),
                    }
                ]
                aggregate = _aggregate_trials(model, backend, config, trials, started_at)
                # Keep the old failure-row shape for in-process failure injection;
                # normal subprocess diagnostics use the richer v0.1.1 shape.
                aggregate["results"] = [
                    {key: row[key] for key in ("batch_size", "status", "reason")}
                    for row in aggregate["results"]
                ]
                return aggregate
    with tempfile.TemporaryDirectory(prefix="edgebench-trials-") as tmp:
        temp = Path(tmp)
        for trial_index in range(config.trials):
            result_path = temp / f"trial-{trial_index}.json"
            command = _child_command(model, backend, config, trial_index, result_path)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(Path.cwd()),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                trials.append(
                    _failed_trial(
                        model,
                        backend,
                        config,
                        trial_index,
                        "timeout",
                        f"Trial exceeded timeout of {config.timeout_seconds:g}s",
                        diagnostic={
                            "stdout": (exc.stdout or "")[-10000:],
                            "stderr": (exc.stderr or "")[-10000:],
                        },
                    )
                )
                continue
            if completed.returncode != 0:
                trials.append(
                    _failed_trial(
                        model,
                        backend,
                        config,
                        trial_index,
                        "crashed",
                        f"Child exited with code {completed.returncode}",
                        diagnostic={
                            "stdout": completed.stdout[-10000:],
                            "stderr": completed.stderr[-10000:],
                        },
                    )
                )
                continue
            if not result_path.exists():
                trials.append(
                    _failed_trial(
                        model,
                        backend,
                        config,
                        trial_index,
                        "missing_result",
                        "Child exited successfully without writing a result",
                        diagnostic={
                            "stdout": completed.stdout[-10000:],
                            "stderr": completed.stderr[-10000:],
                        },
                    )
                )
                continue
            try:
                trial = read_json(result_path)
            except (OSError, json.JSONDecodeError) as exc:
                trial = _failed_trial(
                    model, backend, config, trial_index, "invalid_result", str(exc)
                )
            if completed.stderr:
                trial.setdefault("diagnostic", {})["stderr"] = completed.stderr[-10000:]
            trials.append(trial)
    return _aggregate_trials(model, backend, config, trials, started_at)
