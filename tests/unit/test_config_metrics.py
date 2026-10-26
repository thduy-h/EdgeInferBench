from dataclasses import asdict

import numpy as np
import pytest

from edge_infer_bench.benchmark import latency_metrics
from edge_infer_bench.config import BenchmarkConfig, InputSpec, check_tolerances, parse_batch_sizes
from edge_infer_bench.utils import make_input, read_json, write_json


@pytest.mark.parametrize("text", ["", "0", "-1,4", "1,1", "1,x", "1,"])
def test_invalid_batches(text):
    with pytest.raises(ValueError):
        parse_batch_sizes(text)


def test_batch_parsing():
    assert parse_batch_sizes("1, 4,8,16") == (1, 4, 8, 16)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_iterations": -1},
        {"measurement_iterations": 0},
        {"cpu_threads": 0},
        {"batch_sizes": ()},
        {"random_seed": -1},
        {"random_seed": 2**32},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_latency_percentiles_and_throughput():
    values = [1_000_000, 2_000_000, 3_000_000, 4_000_000]
    result = latency_metrics(values, 4)
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["p99_ms"] == pytest.approx(3.97)
    assert result["mean_ms"] == 2.5
    assert result["std_ms"] == pytest.approx(np.std([1, 2, 3, 4]))
    assert result["throughput_samples_s"] == 1600
    assert result["measured_duration_s"] == 0.01


@pytest.mark.parametrize("durations,batch", [([], 1), ([0], 1), ([-1], 1), ([1], 0)])
def test_invalid_metrics(durations, batch):
    with pytest.raises(ValueError):
        latency_metrics(durations, batch)


def test_deterministic_input():
    spec = InputSpec(height=8, width=8)
    a = make_input(spec, 4, 42)
    assert np.array_equal(a, make_input(spec, 4, 42))
    assert not np.array_equal(a, make_input(spec, 4, 43))
    assert not np.array_equal(a, make_input(spec, 4, 42, 1))
    assert a.shape == (4, 3, 8, 8) and a.dtype == np.float32


@pytest.mark.parametrize("a,r", [(-1, 0), (0, -1), (float("nan"), 0), (0, float("inf"))])
def test_invalid_tolerances(a, r):
    with pytest.raises(ValueError):
        check_tolerances(a, r)


def test_json_roundtrip_and_nonfinite_rejection(tmp_path):
    target = tmp_path / "nested" / "settings.json"
    value = asdict(InputSpec())
    write_json(target, value)
    assert read_json(target) == value
    with pytest.raises(ValueError):
        write_json(target, {"bad": float("nan")})
    assert read_json(target) == value  # Failure did not corrupt the previous result.
