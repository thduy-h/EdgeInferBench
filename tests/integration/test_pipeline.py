import csv
import subprocess
from copy import deepcopy
from html.parser import HTMLParser

import numpy as np
import pytest
from typer.testing import CliRunner

from edge_infer_bench.backends import create_backend
from edge_infer_bench.benchmark import run_benchmark
from edge_infer_bench.cli import app
from edge_infer_bench.config import BenchmarkConfig
from edge_infer_bench.errors import BackendUnavailable, EdgeBenchError
from edge_infer_bench.reporting import comparison_key, generate_report
from edge_infer_bench.utils import read_json, write_json
from edge_infer_bench.validation import validate

pytestmark = pytest.mark.integration


class ResourceParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "link", "img", "iframe"}:
            for name, value in attrs:
                if name in {"src", "href"}:
                    assert not value.startswith(("http:", "https:", "//"))


def test_export_exact_reference_and_dynamic_inference(exported):
    import onnx

    target, reference, metadata = exported
    onnx.checker.check_model(str(target), full_check=True)
    graph = onnx.load(str(target))
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    result = validate(reference, target, "onnxruntime", (1, 4, 8, 16), inputs=2, threads=2)
    assert result["passed"]
    assert len(result["cases"]) == 8
    assert result["source_id"] == metadata["source_id"]
    assert all(case["outputs"][0]["reference_dtype"] == "float32" for case in result["cases"])


def test_real_timing_and_all_report_formats(exported, tmp_path):
    target, _, _ = exported
    config = BenchmarkConfig(1, 5, (1, 4), 2, 42)
    result = run_benchmark(target, "onnxruntime", config)
    assert result["status"] == "ok"
    for row in result["results"]:
        assert row["min_ms"] > 0
        assert row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"] <= row["max_ms"]
        assert row["throughput_samples_s"] > 0
        assert row["peak_observed_rss_mb"] > 0 and row["memory_samples"] >= 2
        assert row["processed_samples"] == row["batch_size"] * 5
        assert len(row["latencies_ns"]) == 5
    original = tmp_path / "results" / "ort.json"
    write_json(original, result)
    for format in ("json", "csv", "html"):
        generate_report(original.parent, tmp_path / f"report.{format}", format)
    assert read_json(tmp_path / "report.json")["runs"][0]["results"] == result["results"]
    with (tmp_path / "report.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert float(rows[0]["p50_ms"]) == result["results"][0]["p50_ms"]
    html = (tmp_path / "report.html").read_text()
    assert "Peak observed process RSS" in html and "onnxruntime" in html
    ResourceParser().feed(html)
    same = deepcopy(result)
    assert comparison_key(result, result["results"][0]) == comparison_key(same, same["results"][0])
    same["config"]["cpu_threads"] += 1
    assert comparison_key(result, result["results"][0]) != comparison_key(same, same["results"][0])


def test_trials_are_isolated_and_aggregated(exported):
    """Two real child interpreters leave raw trials and a reproducible aggregate."""
    config = BenchmarkConfig(0, 2, (1,), 2, 123, trials=2, timeout_seconds=30)
    result = run_benchmark(exported[0], "onnxruntime", config)

    assert result["schema_version"] == 2
    assert result["trial_count"] == 2
    assert result["completed_trials"] == 2
    assert result["failed_trials"] == 0
    assert result["execution_order"] == [0, 1]
    assert len(result["trials"]) == 2
    assert all(trial["status"] == "ok" for trial in result["trials"])
    pids = {trial["process_pid"] for trial in result["trials"]}
    assert len(pids) == 2
    hashes = {trial["results"][0]["input_sha256"] for trial in result["trials"]}
    assert len(hashes) == 1
    row = result["results"][0]
    assert row["completed_trials"] == 2
    assert len(row["trial_p50_ms"]) == 2
    # Fewer than three independent trials are intentionally reported without
    # a confidence interval; the raw trial summaries remain available.
    assert row["p50_ci95_ms"] is None
    assert row["incremental_rss_mb"] >= 0


def test_child_timeout_is_recorded_without_measurements(exported):
    config = BenchmarkConfig(0, 1, (1,), 2, 42, trials=1, timeout_seconds=0.001)
    result = run_benchmark(exported[0], "onnxruntime", config)

    assert result["status"] == "incomplete"
    assert result["failed_trials"] == 1
    trial = result["trials"][0]
    assert trial["status"] == "timeout"
    assert "timeout" in trial["reason"].lower()
    assert "p50_ms" not in result["results"][0]


def test_child_backend_error_is_retained(exported):
    config = BenchmarkConfig(0, 1, (1,), 2, 42, trials=1, timeout_seconds=30)
    result = run_benchmark(exported[0], "does-not-exist", config)

    assert result["status"] == "incomplete"
    assert result["trials"][0]["status"] == "failed"
    assert "Unknown backend" in result["trials"][0]["reason"]
    assert result["results"][0]["status"] == "failed"


def test_schema_one_report_remains_readable(exported, tmp_path):
    config = BenchmarkConfig(0, 1, (1,), 2, 42)
    current = run_benchmark(exported[0], "onnxruntime", config)
    legacy = deepcopy(current)
    legacy["schema_version"] = 1
    legacy["tool_version"] = "0.1.0"
    for field in (
        "trial_count",
        "completed_trials",
        "failed_trials",
        "execution_order",
        "trials",
        "aggregation",
    ):
        legacy.pop(field, None)
    path = tmp_path / "legacy.json"
    write_json(path, legacy)
    report = generate_report(path, tmp_path / "legacy.html", "html")
    assert report["runs"][0]["schema_version"] == 1
    assert "EdgeInferBench / v0.1.1" in (tmp_path / "legacy.html").read_text()


def test_nonzero_child_exit_is_retained(exported, monkeypatch):
    from edge_infer_bench import benchmark as benchmark_module

    def crash(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 7, "child log", "native failure")

    monkeypatch.setattr(benchmark_module.subprocess, "run", crash)
    result = run_benchmark(
        exported[0], "onnxruntime", BenchmarkConfig(0, 1, (1,), 2, 42, timeout_seconds=30)
    )
    assert result["trials"][0]["status"] == "crashed"
    assert result["trials"][0]["diagnostic"]["stderr"] == "native failure"
    assert "Child exited with code 7" in result["trials"][0]["reason"]


def test_wrong_reference_is_rejected(exported, tmp_path):
    from edge_infer_bench.models import REGISTRY, load_model, save_reference

    target, _, _ = exported
    reference = tmp_path / "different"
    save_reference(
        load_model("tiny", 123), reference, "tiny", 123, False, REGISTRY["tiny"].input_spec
    )
    with pytest.raises(EdgeBenchError, match="weights do not match"):
        validate(reference, target, "onnxruntime", (1,), inputs=1)


def test_numerical_failure_nonzero_exit(exported, monkeypatch, tmp_path):
    from edge_infer_bench.backends import ONNXRuntimeBackend

    target, reference, _ = exported
    infer = ONNXRuntimeBackend.infer
    monkeypatch.setattr(
        ONNXRuntimeBackend,
        "infer",
        lambda self, data: [a + np.float32(1) for a in infer(self, data)],
    )
    output = tmp_path / "failure.json"
    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--backend",
            "onnxruntime",
            "--batch-sizes",
            "1",
            "--inputs",
            "1",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert not read_json(output)["passed"]
    assert "FAIL" in result.output
    assert "Error: 1" not in result.output


def test_fixed_batch_actionable_failure(tmp_path):
    from edge_infer_bench.conversion import export_onnx

    path = tmp_path / "fixed.onnx"
    export_onnx("tiny", path, batch_size=1, dynamic_batch=False, threads=2)
    result = run_benchmark(path, "onnxruntime", BenchmarkConfig(0, 2, (1, 4), 2))
    assert result["results"][0]["status"] == "ok"
    assert result["results"][1]["status"] == "failed"
    assert "--dynamic-batch" in result["results"][1]["reason"]


def test_optional_backend_failure_is_recorded(exported, monkeypatch):
    from edge_infer_bench.backends import OpenVINOBackend

    def unavailable(*args):
        raise BackendUnavailable("test runtime unavailable")

    monkeypatch.setattr(OpenVINOBackend, "load", unavailable)
    result = run_benchmark(exported[0], "openvino", BenchmarkConfig(0, 1, (1,), 2))
    assert result["status"] == "incomplete"
    assert result["results"] == [
        {"batch_size": 1, "status": "skipped", "reason": "test runtime unavailable"}
    ]


def test_artifact_tampering_fails_before_inference(exported, tmp_path):
    import shutil

    from edge_infer_bench.artifacts import sidecar

    target, _, _ = exported
    changed = tmp_path / target.name
    shutil.copyfile(target, changed)
    shutil.copyfile(sidecar(target), sidecar(changed))
    with changed.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(EdgeBenchError, match="checksum"):
        create_backend("onnxruntime", changed, 2)


def test_report_failure_rows_and_html_escaping(exported, tmp_path, monkeypatch):
    from edge_infer_bench.backends import OpenVINOBackend

    def unavailable(*args):
        raise BackendUnavailable('<script>alert("bad")</script>')

    monkeypatch.setattr(OpenVINOBackend, "load", unavailable)
    result = run_benchmark(exported[0], "openvino", BenchmarkConfig(0, 1, (1,), 2))
    write_json(tmp_path / "failed.json", result)
    generate_report(tmp_path / "failed.json", tmp_path / "report.html", "html")
    html = (tmp_path / "report.html").read_text()
    assert '<script>alert("bad")</script>' not in html
    assert "&lt;script&gt;" in html and "skipped" in html
    generate_report(tmp_path / "failed.json", tmp_path / "report.csv", "csv")
    with (tmp_path / "report.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "skipped"
    assert "p50_ms" not in row


@pytest.mark.openvino
def test_openvino_conversion_and_execution(exported, tmp_path):
    from edge_infer_bench.conversion import convert_openvino

    ov = pytest.importorskip(
        "openvino", reason="OpenVINO not installed; CPU compatibility unverified"
    )
    output = tmp_path / "tiny.xml"
    try:
        core = ov.Core()
        parameter = ov.opset13.parameter([1, 3], ov.Type.f32)
        identity = ov.Model([parameter], [parameter])
        probe = core.compile_model(identity, "CPU", {"INFERENCE_PRECISION_HINT": ov.Type.f32})
        assert np.array_equal(
            list(probe([np.ones((1, 3), dtype=np.float32)]).values())[0],
            np.ones((1, 3), dtype=np.float32),
        )
    except RuntimeError as exc:
        pytest.skip(f"OpenVINO CPU runtime unavailable: {exc}")
    metadata = convert_openvino(exported[0], output, threads=2)
    assert metadata["compress_to_fp16"] is False
    result = validate(exported[1], output, "openvino", (1, 4), inputs=2, threads=2)
    assert result["passed"]
    adapter = create_backend("openvino", output, 2)
    assert "f32" in adapter.metadata()["inference_precision_hint"]
    adapter.close()
    benchmark = run_benchmark(output, "openvino", BenchmarkConfig(1, 3, (1, 4), 2))
    assert benchmark["status"] == "ok"


@pytest.mark.smoke
def test_resnet18_real_model_smoke(tmp_path):
    from edge_infer_bench.conversion import export_onnx

    path = tmp_path / "resnet18.onnx"
    metadata = export_onnx("resnet18", path, threads=2)
    assert metadata["validation"]["passed"]
    result = run_benchmark(path, "onnxruntime", BenchmarkConfig(1, 2, (1,), 2))
    assert result["results"][0]["p50_ms"] > 0
