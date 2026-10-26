"""Verified conversion. No artifact is published until graph and inference checks pass."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from edge_infer_bench.artifacts import artifact_files, load_metadata, sidecar
from edge_infer_bench.errors import BackendUnavailable, EdgeBenchError
from edge_infer_bench.models import REGISTRY, load_model, save_reference
from edge_infer_bench.utils import sha256, timestamp, write_json
from edge_infer_bench.validation import validate


def _hashes(path: Path) -> dict[str, str]:
    return {part.name: sha256(part) for part in artifact_files(path)}


def export_onnx(
    name: str,
    output: Path,
    *,
    batch_size: int = 1,
    dynamic_batch: bool = True,
    seed: int = 42,
    pretrained: bool = False,
    threads: int = 4,
) -> dict[str, Any]:
    import onnx
    import torch

    if output.suffix != ".onnx":
        raise ValueError("ONNX output must have .onnx extension")
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(REGISTRY)}")
    if batch_size < 1 or threads < 1:
        raise ValueError("Batch size and threads must be positive")
    reference = output.with_name(output.stem + "_reference")
    for path in (output, sidecar(output), reference):
        if path.exists():
            raise EdgeBenchError(f"Refusing to overwrite {path}; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    model = load_model(name, seed, pretrained)
    spec = REGISTRY[name].input_spec
    with tempfile.TemporaryDirectory(prefix=".edgebench-export-", dir=output.parent) as tmp:
        stage = Path(tmp)
        staged_reference = stage / reference.name
        metadata = save_reference(model, staged_reference, name, seed, pretrained, spec)
        staged_output = stage / output.name
        # Batch=1 can be specialized by torch.export. Trace a nontrivial batch for dynamic export.
        trace_batch = max(2, batch_size) if dynamic_batch else batch_size
        sample = torch.zeros(spec.shape(trace_batch), dtype=torch.float32)
        try:
            torch.onnx.export(
                model,
                (sample,),
                str(staged_output),
                dynamo=True,
                input_names=["input"],
                opset_version=18,
                external_data=False,
                dynamic_shapes=({0: torch.export.Dim("batch", min=1)},) if dynamic_batch else None,
            )
        except Exception as exc:
            raise EdgeBenchError(
                f"Dynamo ONNX export failed: {exc}. Try --fixed-batch for this architecture; "
                "use only its exported batch size, or check PyTorch/torchvision compatibility."
            ) from exc
        onnx.checker.check_model(str(staged_output), full_check=True)
        metadata.update(
            {
                "format": "onnx",
                "dynamic_batch": dynamic_batch,
                "batch_size": batch_size,
                "trace_batch_size": trace_batch,
                "opset": 18,
                "exporter": "torch.onnx/dynamo",
                "reference": reference.name,
                "artifact_hashes": _hashes(staged_output),
            }
        )
        write_json(sidecar(staged_output), metadata)
        checked_batches = (1, 4) if dynamic_batch else (batch_size,)
        verification = validate(
            staged_reference,
            staged_output,
            "onnxruntime",
            checked_batches,
            inputs=2,
            seed=seed,
            threads=threads,
        )
        if not verification["passed"]:
            raise EdgeBenchError("ONNX numerical verification failed; no export published")
        verification["reference"] = str(reference)
        verification["target"] = str(output)
        metadata["validation"] = verification
        write_json(sidecar(staged_output), metadata)
        shutil.move(str(staged_reference), str(reference))
        os.replace(staged_output, output)
        os.replace(sidecar(staged_output), sidecar(output))
    return metadata


def convert_openvino(source: Path, output: Path, *, threads: int = 4) -> dict[str, Any]:
    import onnx

    if source.suffix != ".onnx" or output.suffix != ".xml":
        raise ValueError("Conversion requires an .onnx input and an .xml output")
    if threads < 1:
        raise ValueError("Threads must be positive")
    metadata = load_metadata(source)
    onnx.checker.check_model(str(source), full_check=True)
    try:
        import openvino as ov
    except (ImportError, OSError) as exc:
        raise BackendUnavailable(
            f"OpenVINO unavailable: {exc}; install edge-infer-bench[openvino]. "
            "AMD CPU support is not assumed."
        ) from exc
    for part in [*artifact_files(output), sidecar(output)]:
        if part.exists():
            raise EdgeBenchError(f"Refusing to overwrite {part}")
    reference = (source.parent / metadata["reference"]).resolve()
    if not reference.is_dir():
        raise EdgeBenchError("Original reference bundle missing; retain it beside the ONNX export")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".edgebench-ov-", dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        converted = ov.convert_model(str(source))
        ov.save_model(converted, str(staged), compress_to_fp16=False)
        metadata = {
            **metadata,
            "format": "openvino",
            "conversion_time": timestamp(),
            "openvino_version": ov.__version__,
            "compress_to_fp16": False,
            "source_onnx_sha256": sha256(source),
            "artifact_hashes": _hashes(staged),
            "reference": os.path.relpath(reference, output.parent.resolve()),
        }
        metadata.pop("validation", None)
        write_json(sidecar(staged), metadata)
        batches = (1, 4) if metadata["dynamic_batch"] else (metadata["batch_size"],)
        verification = validate(
            reference, staged, "openvino", batches, inputs=2, seed=metadata["seed"], threads=threads
        )
        if not verification["passed"]:
            raise EdgeBenchError("OpenVINO FP32 numerical verification failed; no IR published")
        verification["target"] = str(output)
        metadata["validation"] = verification
        write_json(sidecar(staged), metadata)
        for part in [*artifact_files(staged), sidecar(staged)]:
            os.replace(part, output.parent / part.name)
    return metadata
