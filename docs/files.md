# File inventory

The v0.1.1 maintenance release extends the existing v0.1.0 project. Existing
measurements are retained; no unrelated files were modified.

## Packaging and developer tooling

- [pyproject.toml](../pyproject.toml)
- [flake.nix](../flake.nix)
- [.gitignore](../.gitignore)
- [.github/workflows/ci.yml](../.github/workflows/ci.yml)
- [LICENSE](../LICENSE)
- [CHANGELOG.md](../CHANGELOG.md)

## Application source

- [src/edge_infer_bench/__init__.py](../src/edge_infer_bench/__init__.py)
- [src/edge_infer_bench/artifacts.py](../src/edge_infer_bench/artifacts.py)
- [src/edge_infer_bench/backends.py](../src/edge_infer_bench/backends.py)
- [src/edge_infer_bench/benchmark.py](../src/edge_infer_bench/benchmark.py)
- [src/edge_infer_bench/benchmark_worker.py](../src/edge_infer_bench/benchmark_worker.py)
- [src/edge_infer_bench/cli.py](../src/edge_infer_bench/cli.py)
- [src/edge_infer_bench/config.py](../src/edge_infer_bench/config.py)
- [src/edge_infer_bench/conversion.py](../src/edge_infer_bench/conversion.py)
- [src/edge_infer_bench/environment.py](../src/edge_infer_bench/environment.py)
- [src/edge_infer_bench/errors.py](../src/edge_infer_bench/errors.py)
- [src/edge_infer_bench/models.py](../src/edge_infer_bench/models.py)
- [src/edge_infer_bench/reporting.py](../src/edge_infer_bench/reporting.py)
- [src/edge_infer_bench/schemas.py](../src/edge_infer_bench/schemas.py)
- [src/edge_infer_bench/utils.py](../src/edge_infer_bench/utils.py)
- [src/edge_infer_bench/validation.py](../src/edge_infer_bench/validation.py)
- [src/edge_infer_bench/templates/report.html](../src/edge_infer_bench/templates/report.html)

## Automated tests

- [tests/conftest.py](../tests/conftest.py)
- [tests/unit/test_config_metrics.py](../tests/unit/test_config_metrics.py)
- [tests/unit/test_models_cli.py](../tests/unit/test_models_cli.py)
- [tests/unit/test_validation.py](../tests/unit/test_validation.py)
- [tests/integration/test_pipeline.py](../tests/integration/test_pipeline.py)

## Documentation and examples

- [README.md](../README.md)
- [docs/benchmark-methodology.md](../docs/benchmark-methodology.md)
- [docs/reproducibility.md](../docs/reproducibility.md)
- [docs/openvino-compatibility.md](../docs/openvino-compatibility.md)
- [docs/development-results.md](../docs/development-results.md)
- [docs/python-api.md](../docs/python-api.md)
- [docs/files.md](../docs/files.md)
- [examples/benchmark_cpu.sh](../examples/benchmark_cpu.sh)
- [examples/benchmark_v011_smoke.sh](../examples/benchmark_v011_smoke.sh)

## v0.1.1 retained smoke evidence

- [examples/measured/v011-smoke/README.md](../examples/measured/v011-smoke/README.md)
- [examples/measured/v011-smoke/onnxruntime/benchmark.json](../examples/measured/v011-smoke/onnxruntime/benchmark.json)
- [examples/measured/v011-smoke/onnxruntime/validation.json](../examples/measured/v011-smoke/onnxruntime/validation.json)
- [examples/measured/v011-smoke/openvino/benchmark.json](../examples/measured/v011-smoke/openvino/benchmark.json)
- [examples/measured/v011-smoke/openvino/validation.json](../examples/measured/v011-smoke/openvino/validation.json)

## Actual retained experiments

- [examples/measured/ryzen4800h/README.md](../examples/measured/ryzen4800h/README.md)
- [examples/measured/ryzen4800h/benchmark.csv](../examples/measured/ryzen4800h/benchmark.csv)
- [examples/measured/ryzen4800h/benchmark.html](../examples/measured/ryzen4800h/benchmark.html)
- [examples/measured/ryzen4800h/benchmark.json](../examples/measured/ryzen4800h/benchmark.json)
- [examples/measured/ryzen4800h/environment-freeze.txt](../examples/measured/ryzen4800h/environment-freeze.txt)
- [examples/measured/ryzen4800h/environment.json](../examples/measured/ryzen4800h/environment.json)
- [examples/measured/ryzen4800h/measurements/mobilenet_v2_onnxruntime.json](../examples/measured/ryzen4800h/measurements/mobilenet_v2_onnxruntime.json)
- [examples/measured/ryzen4800h/measurements/mobilenet_v2_pytorch.json](../examples/measured/ryzen4800h/measurements/mobilenet_v2_pytorch.json)
- [examples/measured/ryzen4800h/measurements/mobilenet_v2_validation.json](../examples/measured/ryzen4800h/measurements/mobilenet_v2_validation.json)
- [examples/measured/ryzen4800h/measurements/openvino_unavailable.json](../examples/measured/ryzen4800h/measurements/openvino_unavailable.json)
- [examples/measured/ryzen4800h/measurements/resnet50_onnxruntime.json](../examples/measured/ryzen4800h/measurements/resnet50_onnxruntime.json)
- [examples/measured/ryzen4800h/measurements/resnet50_pytorch.json](../examples/measured/ryzen4800h/measurements/resnet50_pytorch.json)
- [examples/measured/ryzen4800h/measurements/resnet50_validation.json](../examples/measured/ryzen4800h/measurements/resnet50_validation.json)
- [examples/measured/ryzen4800h/openvino-conversion.txt](../examples/measured/ryzen4800h/openvino-conversion.txt)
- [examples/measured/ryzen4800h/verification.txt](../examples/measured/ryzen4800h/verification.txt)

## Local generated assets (ignored by Git)

- `.venv/`: isolated Python environment with local copies of offline dependencies.
- `artifacts/tiny.onnx`, its sidecar, and `artifacts/tiny_reference/`: actual CLI export check.
- `artifacts/development-resnet50/`: actual ResNet50 ONNX, reference state, metadata, results, and reports.
- `artifacts/development-mobilenet/`: actual MobileNetV2 ONNX, reference state, metadata, results, and reports.
- `dist/edge_infer_bench-0.1.0-py3-none-any.whl`: built installable wheel.
- `dist/edge_infer_bench-0.1.0.tar.gz`: built source archive.

Runtime model binaries are deliberately excluded from release sources; their actual metrics/provenance are retained in the experiment directory. Pytest temporary files and diagnostic browser profiles live under `/tmp`, outside release sources.

## v0.1.1 standard measurements

- [examples/measured/v011-standard/README.md](../examples/measured/v011-standard/README.md)
- [examples/measured/v011-standard/benchmark.json](../examples/measured/v011-standard/benchmark.json)
- [examples/measured/v011-standard/benchmark.csv](../examples/measured/v011-standard/benchmark.csv)
- [examples/measured/v011-standard/benchmark.html](../examples/measured/v011-standard/benchmark.html)
