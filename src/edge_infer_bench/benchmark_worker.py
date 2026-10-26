"""Private subprocess entry point for one isolated benchmark trial."""

import sys
from pathlib import Path
from time import perf_counter

from edge_infer_bench.benchmark import _run_trial
from edge_infer_bench.config import BenchmarkConfig
from edge_infer_bench.errors import BackendUnavailable
from edge_infer_bench.utils import read_json, write_json


def main() -> int:
    if len(sys.argv) != 2:
        print("benchmark_worker expects one request JSON path", file=sys.stderr)
        return 2
    request_path = Path(sys.argv[1])
    request = read_json(request_path)
    result_path = Path(request["result_path"])
    model = Path(request["model"])
    config = BenchmarkConfig(**request["config"])
    started = perf_counter()
    try:
        result = _run_trial(model, request["backend"], config, int(request["trial_index"]))
        result["kind"] = "benchmark_trial"
        statuses = [row.get("status") for row in result.get("results", [])]
        if statuses and all(status == "ok" for status in statuses):
            result["status"] = "ok"
        elif statuses and all(status == "skipped" for status in statuses):
            result["status"] = "skipped"
            result["reason"] = next(
                (row.get("reason", "Backend unavailable") for row in result["results"]),
                "Backend unavailable",
            )
        else:
            result["status"] = "partial" if any(status == "ok" for status in statuses) else "failed"
            result.setdefault(
                "reason",
                "; ".join(
                    str(row.get("reason", "batch failed"))
                    for row in result.get("results", [])
                    if row.get("status") != "ok"
                )
                or "Trial failed",
            )
        result["trial_duration_s"] = perf_counter() - started
    except BackendUnavailable as exc:
        result = {
            "schema_version": 2,
            "kind": "benchmark_trial",
            "status": "skipped",
            "trial_index": int(request["trial_index"]),
            "backend": request["backend"],
            "model": str(model),
            "config": request["config"],
            "reason": str(exc),
        }
    except Exception as exc:  # serialized for parent diagnostics; never fabricate metrics
        result = {
            "schema_version": 2,
            "kind": "benchmark_trial",
            "status": "failed",
            "trial_index": int(request["trial_index"]),
            "backend": request["backend"],
            "model": str(model),
            "config": request["config"],
            "reason": str(exc),
            "exception_type": type(exc).__name__,
        }
    write_json(result_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
