# OpenVINO compatibility investigation

OpenVINO is an optional backend. Core installation and ONNX Runtime benchmarking do
not require it. The project treats installation, conversion, and execution as three
separate checks:

1. `import openvino`, version, and `Core().available_devices`.
2. Conversion of a checked ONNX model to FP32 XML/BIN.
3. CPU compilation, inference, and numerical validation in an isolated process.

A package install or device listing alone is not evidence that a model executes on a
machine. `edgebench info` reports availability; `edgebench convert` performs the
actual conversion and CPU verification. Benchmark failures remain explicit.

## v0.1.1 host result

On the AMD Ryzen 7 4800H development host, an isolated environment installed
`openvino==2026.4.1-22982-e213a147257-releases/2026/4`. `ov.Core().available_devices`
returned `['CPU']`. A tiny checked ONNX model was converted to FP32 IR, compiled on
`CPU`, inferred successfully, and compared with the exact PyTorch reference at
batches 1 and 4. The OpenVINO smoke benchmark completed three isolated trials with
three successful batch-1 rows per trial. This establishes a **PASS for the tested
model/runtime path on this machine**; it does not prove universal AMD support or
predict performance for ResNet/MobileNet. The retained outputs are in
[`examples/measured/v011-smoke`](../examples/measured/v011-smoke).

The optional integration test remains conditional and skips when the dependency or
CPU plugin is unavailable. Runtime failures are classified as `PASS`, `FAIL`,
`SKIPPED_DEPENDENCY`, `BLOCKED_ENVIRONMENT`, `UNSUPPORTED_HARDWARE` only when
supported by diagnostic evidence, or `NOT_TESTED`. A missing package is never
reported as hardware incompatibility.

Try the isolated optional environment without replacing a working environment:

```bash
uv venv --python python3.11 /tmp/edgebench-openvino
uv pip install --python /tmp/edgebench-openvino/bin/python -e '.[openvino,dev]'
/tmp/edgebench-openvino/bin/python -c 'import openvino as ov; print(ov.__version__, ov.Core().available_devices)'
edgebench convert --input artifacts/resnet18.onnx --format openvino \
  --output artifacts/resnet18.xml
edgebench validate --reference artifacts/resnet18_reference \
  --target artifacts/resnet18.xml --backend openvino --batch-sizes 1,4
```

Native-library or daemon restrictions can block installation independently of CPU
compatibility. Record the exact command and diagnostic instead of relabeling that
condition as unsupported hardware.
