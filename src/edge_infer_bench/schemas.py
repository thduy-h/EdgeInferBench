"""JSON wire contracts for benchmark and validation results.

Missing measurements are absent, never zero-filled.  Benchmark aggregates use
schema 2 while the individual trial payload retains the same batch-row shape
for convenient inspection and v0.1.0 compatibility.
"""

from typing import Any, Literal, NotRequired, TypedDict


class LatencyMetrics(TypedDict):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_samples_s: float
    measured_duration_s: float


class BatchResult(TypedDict):
    batch_size: int
    status: Literal["ok", "failed", "skipped"]
    reason: NotRequired[str]
    input_shape: NotRequired[list[int]]
    p50_ms: NotRequired[float]
    p95_ms: NotRequired[float]
    p99_ms: NotRequired[float]
    mean_ms: NotRequired[float]
    std_ms: NotRequired[float]
    min_ms: NotRequired[float]
    max_ms: NotRequired[float]
    throughput_samples_s: NotRequired[float]
    measured_duration_s: NotRequired[float]
    rss_mb: NotRequired[float]
    peak_observed_rss_mb: NotRequired[float]
    memory_samples: NotRequired[int]
    memory_sample_interval_ms: NotRequired[int]
    first_inference_ms: NotRequired[float]
    latencies_ns: NotRequired[list[int]]
    processed_samples: NotRequired[int]
    model_artifact_mb: NotRequired[float]
    baseline_rss_mb: NotRequired[float]
    rss_after_initialization_mb: NotRequired[float]
    rss_after_benchmark_mb: NotRequired[float]
    incremental_rss_mb: NotRequired[float]
    peak_incremental_rss_mb: NotRequired[float]
    input_sha256: NotRequired[str]


class BenchmarkResult(TypedDict):
    schema_version: Literal[1, 2]
    kind: Literal["benchmark"]
    created_at: str
    tool_version: str
    backend: str
    model: str
    model_metadata: dict[str, Any]
    config: dict[str, Any]
    backend_settings: dict[str, Any]
    initialization_ms: float | None
    baseline_before_load_rss_mb: float
    environment: dict[str, Any]
    status: Literal["ok", "incomplete"]
    results: list[BatchResult]
    timing_scope: str
    trial_count: NotRequired[int]
    completed_trials: NotRequired[int]
    failed_trials: NotRequired[int]
    execution_order: NotRequired[list[int]]
    trials: NotRequired[list[dict[str, Any]]]
    aggregation: NotRequired[dict[str, Any]]


class TensorComparison(TypedDict):
    index: int
    reference_shape: list[int]
    target_shape: list[int]
    reference_dtype: str
    target_dtype: str
    max_absolute_error: float | None
    mean_absolute_error: float | None
    max_relative_error: float | None
    mean_relative_error: float | None
    mismatched_elements: int | None
    allclose_passed: bool
    atol: float
    rtol: float
    reason: NotRequired[str]


class ComparisonResult(TypedDict):
    passed: bool
    reference_output_count: int
    target_output_count: int
    mismatched_output_count: int
    reference_shapes: list[list[int]]
    target_shapes: list[list[int]]
    outputs: list[TensorComparison]
    relative_error_denominator: str
    atol: float
    rtol: float


class ValidationCase(ComparisonResult):
    batch_size: int
    input_index: int


class ValidationResult(TypedDict):
    schema_version: Literal[1]
    kind: Literal["validation"]
    created_at: str
    passed: bool
    backend: str
    source_id: str
    reference: str
    target: str
    seed: int
    batch_sizes: list[int]
    inputs_per_batch: int
    atol: float
    rtol: float
    backend_settings: dict[str, Any]
    cases: list[ValidationCase]
