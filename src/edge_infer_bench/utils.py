"""Small I/O and reproducibility primitives. JSON never permits NaN/Infinity."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from edge_infer_bench.config import InputSpec


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    atomic_text(path, text)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_input(spec: InputSpec, batch: int, seed: int, case: int = 0) -> np.ndarray:
    # Independent streams avoid dependence on batch ordering or previous calls.
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch, case]))
    return rng.standard_normal(spec.shape(batch), dtype=np.float32)
