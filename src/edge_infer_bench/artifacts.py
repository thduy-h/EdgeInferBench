"""Artifact integrity and input contract shared by all adapters."""

from pathlib import Path
from typing import Any

from edge_infer_bench.config import InputSpec
from edge_infer_bench.errors import EdgeBenchError
from edge_infer_bench.utils import read_json, sha256


def sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def artifact_files(path: Path) -> list[Path]:
    if path.is_dir():
        return [path / "state.pt", path / "metadata.json"]
    return [path, path.with_suffix(".bin")] if path.suffix == ".xml" else [path]


def artifact_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in artifact_files(path)) / 1_000_000


def load_metadata(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return read_json(path / "metadata.json")
    metadata = read_json(sidecar(path))
    if metadata.get("schema_version") != 1:
        raise EdgeBenchError("Unsupported artifact metadata schema")
    for part in artifact_files(path):
        if metadata["artifact_hashes"].get(part.name) != sha256(part):
            raise EdgeBenchError(f"Artifact checksum mismatch: {part}")
    return metadata


def check_batch(metadata: dict[str, Any], batch: int) -> None:
    if batch < 1:
        raise ValueError("Batch size must be positive")
    if not metadata.get("dynamic_batch", True) and batch != metadata["batch_size"]:
        raise EdgeBenchError(
            f"Fixed-batch artifact expects {metadata['batch_size']}, got {batch}; "
            "re-export with --dynamic-batch or benchmark the exported batch size"
        )


def input_spec(metadata: dict[str, Any]) -> InputSpec:
    return InputSpec(**metadata["input_spec"])
