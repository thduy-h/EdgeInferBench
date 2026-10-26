"""Strict comparison groups, portable JSON/CSV, and self-contained offline HTML."""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from edge_infer_bench.errors import EdgeBenchError
from edge_infer_bench.utils import atomic_text, read_json, timestamp, write_json


def collect_results(source: Path) -> dict[str, Any]:
    paths = sorted(source.rglob("*.json")) if source.is_dir() else [source]
    runs, validations = [], []
    seen: set[str] = set()
    for path in paths:
        value = read_json(path)
        kind = value.get("kind")
        if kind not in {"report", "benchmark", "validation"}:
            if source.is_dir():
                continue
            raise EdgeBenchError(f"Expected benchmark/validation result, got {path}")
        if kind == "report":
            candidates = value["runs"] + value.get("validations", [])
        else:
            candidates = [value]
        for candidate in candidates:
            digest = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            if candidate.get("schema_version") not in {1, 2}:
                raise EdgeBenchError(f"Unsupported result schema in {path}")
            if candidate.get("kind") == "benchmark":
                if not candidate.get("results") or "model_metadata" not in candidate:
                    raise EdgeBenchError(f"Invalid benchmark result in {path}")
                runs.append(candidate)
            elif candidate.get("kind") == "validation":
                validations.append(candidate)
            elif not source.is_dir():
                raise EdgeBenchError(f"Expected benchmark/validation result, got {path}")
    if not runs and not validations:
        raise EdgeBenchError(f"No benchmark or validation results found in {source}")
    return {
        "schema_version": 1,
        "kind": "report",
        "created_at": timestamp(),
        "runs": runs,
        "validations": validations,
    }


def comparison_key(run: dict[str, Any], row: dict[str, Any]) -> str:
    env, model, config = run["environment"], run["model_metadata"], run["config"]
    contract = {
        "source_id": model.get("source_id"),
        "input_spec": model.get("input_spec"),
        "precision": model.get("precision"),
        "batch": row["batch_size"],
        "input_sha256": row.get("input_sha256"),
        "config": {k: v for k, v in config.items() if k != "batch_sizes"},
        "environment": {
            k: env.get(k)
            for k in (
                "cpu",
                "architecture",
                "os",
                "distribution",
                "python",
                "logical_cpus",
                "cpu_affinity",
                "governors",
                "thread_environment",
                "versions",
            )
        },
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:16]


def report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in report["runs"]:
        metadata = run.get("model_metadata", {})
        environment = run.get("environment", {})
        config = run.get("config", {})
        for result in run["results"]:
            rows.append(
                {
                    "comparison_group": comparison_key(run, result),
                    "backend": run["backend"],
                    "architecture": metadata.get("architecture", "unknown"),
                    "source_id": metadata.get("source_id"),
                    "precision": metadata.get("precision", "unknown"),
                    "cpu": environment.get("cpu", "unknown"),
                    "created_at": run["created_at"],
                    "threads": config.get("cpu_threads"),
                    "warmup": config.get("warmup_iterations"),
                    "iterations": config.get("measurement_iterations"),
                    "seed": config.get("random_seed"),
                    "initialization_ms": run.get("initialization_ms"),
                    "correctness_passed": metadata.get("validation", {}).get("passed"),
                    "backend_settings": json.dumps(run.get("backend_settings", {}), sort_keys=True),
                    "versions": json.dumps(environment.get("versions", {}), sort_keys=True),
                    **{k: v for k, v in result.items() if k != "latencies_ns"},
                }
            )
    return rows


def write_csv(report: dict[str, Any], output: Path) -> None:
    rows = report_rows(report)
    for result in report.get("validations", []):
        rows.append(
            {
                "architecture": "validation",
                "backend": result["backend"],
                "source_id": result["source_id"],
                "correctness_passed": result["passed"],
                "status": "ok" if result["passed"] else "failed",
                "validation_details": json.dumps(result, sort_keys=True),
            }
        )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output, stream.getvalue())


def write_html(report: dict[str, Any], output: Path) -> None:
    rows = report_rows(report)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["comparison_group"], []).append(row)
    charts = []
    for key, members in groups.items():
        successful = [m for m in members if m["status"] == "ok"]
        if not successful:
            continue
        metrics = []
        for field, label, unit in (
            ("p50_ms", "P50 latency", "ms"),
            ("p95_ms", "P95 latency", "ms"),
            ("p99_ms", "P99 latency", "ms"),
            ("throughput_samples_s", "Throughput", "samples/s"),
            ("peak_observed_rss_mb", "Peak observed process RSS", "MB"),
        ):
            maximum = max(r[field] for r in successful)
            metrics.append(
                {
                    "label": label,
                    "unit": unit,
                    "bars": [
                        {
                            "backend": r["backend"],
                            "value": r[field],
                            "width": 100 * r[field] / maximum if maximum else 0,
                        }
                        for r in successful
                    ],
                }
            )
        charts.append(
            {
                "key": key,
                "model": successful[0]["architecture"],
                "batch": successful[0]["batch_size"],
                "metrics": metrics,
            }
        )
    checks = list(report.get("validations", []))
    seen = {json.dumps(check, sort_keys=True) for check in checks}
    for run in report["runs"]:
        check = run["model_metadata"].get("validation")
        if check and json.dumps(check, sort_keys=True) not in seen:
            checks.append(check)
            seen.add(json.dumps(check, sort_keys=True))
    environment = Environment(
        loader=PackageLoader("edge_infer_bench", "templates"),
        autoescape=select_autoescape(["html"]),
    )
    html = environment.get_template("report.html").render(
        report=report,
        rows=rows,
        charts=charts,
        checks=checks,
        successes=sum(r["status"] == "ok" for r in rows),
    )
    atomic_text(output, html)


def generate_report(source: Path, output: Path, format: str) -> dict[str, Any]:
    report = collect_results(source)
    if format == "json":
        write_json(output, report)
    elif format == "csv":
        write_csv(report, output)
    elif format == "html":
        write_html(report, output)
    else:
        raise ValueError("Report format must be json, csv, or html")
    return report
