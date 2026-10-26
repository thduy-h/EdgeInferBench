"""Numerical equivalence, independent of inference runtime."""

from pathlib import Path

import numpy as np

from edge_infer_bench.artifacts import check_batch, input_spec
from edge_infer_bench.backends import create_backend
from edge_infer_bench.config import check_tolerances
from edge_infer_bench.errors import EdgeBenchError
from edge_infer_bench.schemas import (
    ComparisonResult,
    TensorComparison,
    ValidationCase,
    ValidationResult,
)
from edge_infer_bench.utils import make_input, timestamp


def compare_outputs(
    reference: list[np.ndarray], target: list[np.ndarray], atol: float, rtol: float
) -> ComparisonResult:
    check_tolerances(atol, rtol)
    tensors: list[TensorComparison] = []
    for index, (a, b) in enumerate(zip(reference, target, strict=False)):
        item: TensorComparison = {
            "index": index,
            "reference_shape": list(a.shape),
            "target_shape": list(b.shape),
            "reference_dtype": str(a.dtype),
            "target_dtype": str(b.dtype),
            "atol": atol,
            "rtol": rtol,
            "allclose_passed": False,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "max_relative_error": None,
            "mean_relative_error": None,
            "mismatched_elements": None,
        }
        if a.shape != b.shape:
            item["reason"] = "shape_mismatch"
        elif not np.isfinite(a).all() or not np.isfinite(b).all():
            item["reason"] = "non_finite_output"
        elif a.size == 0:
            item["reason"] = "empty_output"
        else:
            dtype = np.complex128 if a.dtype.kind == "c" or b.dtype.kind == "c" else np.float64
            a64, b64 = a.astype(dtype), b.astype(dtype)
            absolute = np.abs(a64 - b64)
            relative = absolute / np.maximum(np.abs(a64), 1e-8)
            item.update(
                {
                    "max_absolute_error": float(absolute.max()),
                    "mean_absolute_error": float(absolute.mean()),
                    "max_relative_error": float(relative.max()),
                    "mean_relative_error": float(relative.mean()),
                    "mismatched_elements": int(
                        np.count_nonzero(
                            ~np.isclose(b64, a64, atol=atol, rtol=rtol, equal_nan=False)
                        )
                    ),
                    "allclose_passed": bool(
                        a.dtype == b.dtype
                        and np.allclose(b64, a64, atol=atol, rtol=rtol, equal_nan=False)
                    ),
                }
            )
            if a.dtype != b.dtype:
                item["reason"] = "dtype_mismatch"
        tensors.append(item)
    return {
        "passed": bool(reference)
        and len(reference) == len(target)
        and all(t["allclose_passed"] for t in tensors),
        "reference_output_count": len(reference),
        "target_output_count": len(target),
        "mismatched_output_count": abs(len(reference) - len(target)),
        "reference_shapes": [list(a.shape) for a in reference],
        "target_shapes": [list(b.shape) for b in target],
        "outputs": tensors,
        "relative_error_denominator": "max(abs(reference), 1e-8)",
        "atol": atol,
        "rtol": rtol,
    }


def validate(
    reference: Path,
    target: Path,
    backend: str,
    batch_sizes: tuple[int, ...],
    *,
    inputs: int = 3,
    seed: int = 42,
    threads: int = 4,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> ValidationResult:
    from edge_infer_bench.config import BenchmarkConfig

    check_tolerances(atol, rtol)
    BenchmarkConfig(batch_sizes=batch_sizes, random_seed=seed, cpu_threads=threads)
    if inputs < 1:
        raise ValueError("Validation inputs must be >= 1")
    source = create_backend("pytorch", reference, threads)
    destination = None
    try:
        destination = create_backend(backend, target, threads)
        if source.model_metadata["source_id"] != destination.model_metadata["source_id"]:
            raise EdgeBenchError("Reference weights do not match the target's exported source_id")
        if source.model_metadata["input_spec"] != destination.model_metadata["input_spec"]:
            raise EdgeBenchError("Reference and target input specifications differ")
        cases: list[ValidationCase] = []
        for batch in batch_sizes:
            check_batch(destination.model_metadata, batch)
            for case in range(inputs):
                data = make_input(input_spec(source.model_metadata), batch, seed, case)
                comparison = compare_outputs(
                    source.infer(source.prepare_input(data)),
                    destination.infer(destination.prepare_input(data)),
                    atol,
                    rtol,
                )
                cases.append({"batch_size": batch, "input_index": case, **comparison})
        return {
            "schema_version": 1,
            "kind": "validation",
            "created_at": timestamp(),
            "passed": all(c["passed"] for c in cases),
            "backend": backend,
            "source_id": source.model_metadata["source_id"],
            "reference": str(reference),
            "target": str(target),
            "seed": seed,
            "batch_sizes": list(batch_sizes),
            "inputs_per_batch": inputs,
            "atol": atol,
            "rtol": rtol,
            "backend_settings": destination.metadata(),
            "cases": cases,
        }
    finally:
        source.close()
        if destination is not None:
            destination.close()
