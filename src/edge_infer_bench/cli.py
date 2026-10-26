"""Thin CLI handlers; engines are also usable as Python APIs."""

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from edge_infer_bench import __version__
from edge_infer_bench.config import BenchmarkConfig, parse_batch_sizes
from edge_infer_bench.errors import EdgeBenchError

app = typer.Typer(no_args_is_help=True, help="Verified conversion and reproducible CPU inference.")
console = Console()
errors = Console(stderr=True)


def friendly(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except typer.Exit:
            raise
        except (EdgeBenchError, ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
            errors.print(f"Error: {exc}", style="red", markup=False)
            raise typer.Exit(1) from exc

    return wrapper


@app.callback(invoke_without_command=True)
def main(version: bool = typer.Option(False, "--version", help="Print version and exit.")) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command()
@friendly
def info(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Inspect hardware, installed versions, providers and devices."""
    from edge_infer_bench.environment import environment_info

    result = environment_info(probe=True)
    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        table = Table(title="EdgeInferBench environment")
        table.add_column("Property")
        table.add_column("Value")
        for key, value in result.items():
            table.add_row(
                key, json.dumps(value, indent=2) if isinstance(value, dict) else str(value)
            )
        console.print(table)


@app.command("export")
@friendly
def export_command(
    model: str = typer.Option(..., help="resnet18, resnet50, mobilenet_v2, or tiny."),
    output: Path = typer.Option(..., help="Destination .onnx; reference bundle saved alongside."),
    format: str = typer.Option("onnx", help="Export format (onnx)."),
    batch_size: int = typer.Option(1, min=1),
    dynamic_batch: bool = typer.Option(True, "--dynamic-batch/--fixed-batch"),
    seed: int = typer.Option(42, min=0, max=2**32 - 1),
    pretrained: bool = typer.Option(False, help="Download torchvision DEFAULT pretrained weights."),
    threads: int = typer.Option(4, min=1),
) -> None:
    """Export with dynamo, check ONNX, and verify against exact source weights."""
    from edge_infer_bench.conversion import export_onnx

    if format != "onnx":
        raise ValueError("Export format must be onnx; use convert for OpenVINO")
    if pretrained:
        console.print("Pretrained weights requested: torchvision may download large weight files.")
    metadata = export_onnx(
        model,
        output,
        batch_size=batch_size,
        dynamic_batch=dynamic_batch,
        seed=seed,
        pretrained=pretrained,
        threads=threads,
    )
    console.print(f"Exported and numerically verified: {output}", style="green")
    console.print(f"Reference: {output.parent / metadata['reference']}")


@app.command()
@friendly
def convert(
    input: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(...),
    format: str = typer.Option("openvino"),
    threads: int = typer.Option(4, min=1),
) -> None:
    """Convert checked ONNX to FP32 OpenVINO IR, then verify real CPU inference."""
    from edge_infer_bench.conversion import convert_openvino

    if format != "openvino":
        raise ValueError("Conversion format must be openvino")
    convert_openvino(input, output, threads=threads)
    console.print(f"Converted and numerically verified on CPU: {output}", style="green")


@app.command()
@friendly
def validate(
    reference: Path = typer.Option(..., exists=True, file_okay=False),
    target: Path = typer.Option(..., exists=True),
    backend: str = typer.Option(...),
    batch_sizes: str = typer.Option("1,4,8"),
    inputs: int = typer.Option(3, min=1),
    atol: float = typer.Option(1e-4, min=0),
    rtol: float = typer.Option(1e-3, min=0),
    seed: int = typer.Option(42, min=0, max=2**32 - 1),
    threads: int = typer.Option(4, min=1),
    output: Path | None = typer.Option(None, help="Save detailed validation JSON."),
) -> None:
    """Compare deterministic outputs; exit 1 for any numerical failure."""
    from edge_infer_bench.utils import write_json
    from edge_infer_bench.validation import validate as validate_model

    result = validate_model(
        reference,
        target,
        backend,
        parse_batch_sizes(batch_sizes),
        inputs=inputs,
        atol=atol,
        rtol=rtol,
        seed=seed,
        threads=threads,
    )
    if output:
        write_json(output, result)
    table = Table(title="Numerical validation")
    for field in (
        "Batch",
        "Input",
        "Output",
        "Max absolute",
        "Mean absolute",
        "Max relative",
        "Pass",
    ):
        table.add_column(field)
    for case in result["cases"]:
        for tensor in case["outputs"]:
            table.add_row(
                str(case["batch_size"]),
                str(case["input_index"]),
                str(tensor["index"]),
                str(tensor.get("max_absolute_error", tensor.get("reason"))),
                str(tensor.get("mean_absolute_error", "—")),
                str(tensor.get("max_relative_error", "—")),
                str(tensor["allclose_passed"]),
            )
    console.print(table)
    console.print(f"{'PASS' if result['passed'] else 'FAIL'} · atol={atol} rtol={rtol}")
    if not result["passed"]:
        raise typer.Exit(1)


@app.command()
@friendly
def benchmark(
    model: Path = typer.Option(
        ..., exists=True, help=".onnx, .xml, or PyTorch reference directory."
    ),
    backend: str = typer.Option(...),
    output: Path = typer.Option(..., help="Benchmark result JSON."),
    mode: str = typer.Option("standard", help="standard (5 trials/20/200) or smoke (3/5/30)."),
    trials: int | None = typer.Option(None, min=1, help="Independent child-process trials."),
    batch_sizes: str = typer.Option("1,4,8,16"),
    warmup: int | None = typer.Option(None, min=0),
    iterations: int | None = typer.Option(None, min=1),
    threads: int = typer.Option(4, min=1),
    seed: int = typer.Option(42, min=0, max=2**32 - 1),
    timeout: float = typer.Option(1800.0, min=1.0, help="Per-trial child timeout in seconds."),
) -> None:
    """Measure synchronous CPU inference in fresh child processes and aggregate trials."""
    from edge_infer_bench.backends import BACKENDS
    from edge_infer_bench.benchmark import run_benchmark
    from edge_infer_bench.utils import write_json

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend; choose from {', '.join(BACKENDS)}")
    if mode not in {"standard", "smoke"}:
        raise ValueError("Mode must be standard or smoke")
    defaults = (5, 30, 3) if mode == "smoke" else (20, 200, 5)
    config = BenchmarkConfig(
        warmup if warmup is not None else defaults[0],
        iterations if iterations is not None else defaults[1],
        parse_batch_sizes(batch_sizes),
        threads,
        seed,
        trials if trials is not None else defaults[2],
        timeout,
    )
    result = run_benchmark(model, backend, config)
    write_json(output, result)
    table = Table(title=f"{backend} CPU inference")
    for field in ("Batch", "Status", "P50 ms", "P95 ms", "P99 ms", "Samples/s", "RSS MB"):
        table.add_column(field)
    for row in result["results"]:
        values = [
            f"{row[k]:.3f}" if k in row else "—"
            for k in (
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "throughput_samples_s",
                "rss_mb",
            )
        ]
        table.add_row(str(row["batch_size"]), row["status"], *values)
        if "reason" in row:
            errors.print(row["reason"], style="red", markup=False)
    console.print(table)
    console.print(
        f"Result saved: {output} · "
        f"{result['completed_trials']}/{result['trial_count']} trials completed"
    )
    if result["status"] != "ok":
        raise typer.Exit(1)


@app.command()
@friendly
def report(
    input: Path = typer.Option(..., exists=True, help="Result JSON or directory (recursive)."),
    output: Path = typer.Option(...),
    format: str = typer.Option("html", help="html, csv, or json."),
) -> None:
    """Generate reports from actual benchmark and validation results."""
    from edge_infer_bench.reporting import generate_report

    generate_report(input, output, format)
    console.print(f"Report saved: {output}")


if __name__ == "__main__":
    app()
