import pytest
from typer.testing import CliRunner

from edge_infer_bench.cli import app
from edge_infer_bench.models import REGISTRY, load_model, load_reference, save_reference

runner = CliRunner()


def test_registry_and_model_determinism(tmp_path):
    import torch

    assert {"resnet18", "resnet50", "mobilenet_v2"} <= REGISTRY.keys()
    first, same, different = [load_model("tiny", seed) for seed in [42, 42, 43]]
    assert torch.equal(first[0].weight, same[0].weight)
    assert not torch.equal(first[0].weight, different[0].weight)
    save_reference(first, tmp_path / "reference", "tiny", 42, False, REGISTRY["tiny"].input_spec)
    restored, _ = load_reference(tmp_path / "reference")
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), restored.parameters(), strict=True)
    )
    checkpoint = tmp_path / "reference" / "state.pt"
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(Exception, match="checksum"):
        load_reference(tmp_path / "reference")


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("not-a-model")


@pytest.mark.parametrize(
    "args", [["--help"], ["export", "--help"], ["benchmark", "--help"], ["--version"]]
)
def test_cli_help(args):
    assert runner.invoke(app, args).exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["benchmark"],
        ["export", "--model", "tiny", "--output", "a.onnx", "--batch-size", "0"],
        ["export", "--model", "tiny", "--output", "a.onnx", "--format", "invalid"],
        ["export", "--model", "missing", "--output", "a.onnx"],
    ],
)
def test_readable_cli_errors(args):
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "Error" in result.output
    assert "Traceback" not in result.output
