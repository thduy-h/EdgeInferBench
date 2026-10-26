"""Extensible model registry and safe, exact state-dict reference bundles."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from edge_infer_bench.config import InputSpec
from edge_infer_bench.errors import EdgeBenchError
from edge_infer_bench.utils import read_json, sha256, timestamp, write_json


@dataclass(frozen=True)
class ModelEntry:
    factory: Callable[[bool], Any]
    input_spec: InputSpec


REGISTRY: dict[str, ModelEntry] = {}


def register_model(name: str, factory: Callable[[bool], Any], spec: InputSpec) -> None:
    if name in REGISTRY:
        raise ValueError(f"Model already registered: {name}")
    REGISTRY[name] = ModelEntry(factory, spec)


def _vision(name: str, pretrained: bool) -> Any:
    import torchvision.models as models

    return models.get_model(name, weights="DEFAULT" if pretrained else None)


def _tiny(pretrained: bool) -> Any:
    import torch.nn as nn

    if pretrained:
        raise ValueError("tiny has no pretrained weights")
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 5),
    )


for _name in ("resnet18", "resnet50", "mobilenet_v2"):
    register_model(_name, lambda pretrained, name=_name: _vision(name, pretrained), InputSpec())
register_model("tiny", _tiny, InputSpec(height=8, width=8))


def load_model(name: str, seed: int = 42, pretrained: bool = False) -> Any:
    import torch

    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(REGISTRY)}")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = REGISTRY[name].factory(pretrained)
    return model.float().cpu().eval()


def save_reference(
    model: Any, directory: Path, name: str, seed: int, pretrained: bool, spec: InputSpec
) -> dict[str, Any]:
    import torch
    import torchvision

    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = directory / "state.pt"
    torch.save(model.state_dict(), checkpoint)
    metadata = {
        "schema_version": 1,
        "architecture": name,
        "seed": seed,
        "weights": "pretrained_DEFAULT" if pretrained else "random",
        "source_id": sha256(checkpoint),
        "input_spec": asdict(spec),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "created_at": timestamp(),
        "precision": "FP32",
        "preprocessing": {
            "benchmark": "Synthetic standard-normal NCHW input; no image preprocessing timed",
            "pretrained_accuracy": "Use torchvision DEFAULT weights.transforms() for real images",
        },
    }
    write_json(directory / "metadata.json", metadata)
    return metadata


def load_reference(directory: Path) -> tuple[Any, dict[str, Any]]:
    import torch
    import torchvision

    metadata = read_json(directory / "metadata.json")
    if metadata.get("schema_version") != 1:
        raise EdgeBenchError("Unsupported reference schema")
    if sha256(directory / "state.pt") != metadata["source_id"]:
        raise EdgeBenchError("Reference checkpoint checksum mismatch; restore the original bundle")
    if metadata["torchvision_version"] != torchvision.__version__:
        raise EdgeBenchError("Reference torchvision version differs; use the recorded environment")
    model = load_model(metadata["architecture"], metadata["seed"], pretrained=False)
    # Restrict deserialization to weights. A checkpoint is still a trusted input.
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval(), metadata
