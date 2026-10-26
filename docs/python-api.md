# Python API and result contracts

The CLI delegates to the same functions used by the integration tests. Example:

```python
from pathlib import Path

from edge_infer_bench.benchmark import run_benchmark
from edge_infer_bench.config import BenchmarkConfig
from edge_infer_bench.conversion import export_onnx
from edge_infer_bench.utils import write_json
from edge_infer_bench.validation import validate

artifact = Path("artifacts/api_tiny.onnx")
export_onnx("tiny", artifact, seed=42, dynamic_batch=True)
reference = artifact.with_name("api_tiny_reference")
verification = validate(reference, artifact, "onnxruntime", (1, 4), inputs=3)
assert verification["passed"]
result = run_benchmark(
    artifact,
    "onnxruntime",
    BenchmarkConfig(
        warmup_iterations=5,
        measurement_iterations=30,
        batch_sizes=(1, 4),
        cpu_threads=4,
    ),
)
write_json(Path("results/api_tiny.json"), result)
assert result["status"] == "ok"
```

Exports intentionally refuse overwrite. Always inspect a returned benchmark's
status: unavailable/failed backend measurements are records, not raised numerical
results. Invalid artifacts/configuration can raise before execution. `validate`
returns `passed=False` for numerical mismatches; the CLI translates this into exit 1.
Missing runtime/provenance/configuration errors raise actionable exceptions.

## Versioned wire formats

`schemas.py` defines TypedDict contracts for measurements and validation. JSON
objects have `schema_version: 1` or `2` and `kind: benchmark`, `benchmark_trial`,
`validation`, or `report`. Schema 2 benchmark aggregates retain independent child
trial objects and an execution order; schema 1 files remain readable by reporting.
Strict serialization rejects NaN/Infinity. Paths are diagnostic provenance, while
the source checkpoint's SHA-256 ties conversions to exact reference weights.

- A benchmark contains `model_metadata`, validated `config`, effective
  `backend_settings`, `environment`, initialization/baseline memory, and per-batch
  `results`. Successful rows have positive measured latencies, raw `latencies_ns`,
  processed samples, observed RSS, and artifact size. Schema 2 also contains
  `trial_count`, completed/failed counts, raw trials, median aggregation metadata,
  between-trial variability, and bootstrap intervals where at least three trials
  complete. Failed/skipped rows have only batch, status, and reason; absent metrics
  are never filled with zero.
- A validation contains the source identity, backend/settings, seed, tolerances,
  tested batches/inputs, and `cases`. Each case records output counts/shapes and
  per-tensor metrics. Non-comparable/non-finite tensors have null error metrics
  and a reason. Output dtype/shape/count mismatches fail even when values overlap.
- A report preserves full benchmark `runs` and standalone `validations`. CSV
  flattens batch rows with comparison group IDs and JSON settings columns; explicit
  validations get a details column. HTML includes grouped charts and provenance.

## Extending a model

Use `models.register_model(name, factory, InputSpec(...))`. The factory accepts a
pretrained boolean and returns an eager CPU PyTorch module. The default loader
controls initialization seed, FP32 dtype, and evaluation mode. Register the same
factory in every restoring process, including any CLI wrapper. Factories must
produce the same architecture for state restoration; downloading pretrained
weights must only occur when requested. v0.1 supports one NCHW float32 tensor input.

## Extending a backend

Implement `InferenceBackend.load`, `prepare_input`, `infer`, `metadata`, and `close`,
then register its class in `backends.BACKENDS`. `load` must establish the artifact's
input/provenance metadata and report initialization time. `infer` must synchronously
complete and return ordered NumPy arrays; do not enqueue unfinished work. Normalize
output trees with `normalize_outputs` when appropriate. Preparing inputs must not
perform hidden work later included differently across backends. Metadata should
record actual precision, device, version, thread/stream settings, and graph policies.
Keep dependency imports local so unavailable adapters do not affect other commands.
