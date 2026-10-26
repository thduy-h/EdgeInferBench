# Development execution report

## v0.1.1 release verification — 27 October 2026

These are actual checks on NixOS x86_64 with an AMD Ryzen 7 4800H (8 physical, 16
logical CPUs), Python 3.11.16. Existing v0.1.0 artifacts under
`examples/measured/ryzen4800h/` were preserved and not overwritten.

- Full suite: **52 passed, 1 skipped**. The skip is the conditional OpenVINO test in
  the original environment, where the package was absent. With the isolated OpenVINO
  environment, the complete suite passed (**53 passed**).
- Ruff lint and format checks: passed.
- CPU-only isolated environment: **passed**. Official CPU wheels installed
  `torch==2.14.1+cpu` and `torchvision==0.29.1+cpu`; the package, export, ONNX
  validation, smoke benchmark, and offline HTML report all executed from `/tmp`.
- OpenVINO isolated environment: **passed for the tested tiny model path** with
  `openvino==2026.4.1` (build `2026.4.1-22982-e213a147257-releases/2026/4`),
  `Core().available_devices == ['CPU']`, FP32 IR conversion,
  CPU compilation, inference, numerical validation, and three child-process trials.
- Nix: `nix-instantiate --parse flake.nix` passed. `nix flake show --no-write-lock-file`
  and `nix develop --no-write-lock-file -c python --version` could not evaluate this
  uncommitted directory because Nix tried to fetch `git+file:///home/thduy-h` and
  reported that `/home/thduy-h` is not a Git repository. No lockfile was fabricated;
  a real daemon-backed Nix checkout remains to be tested. The earlier managed-session
`/nix/var/nix/daemon-socket/socket` denial is an execution-sandbox restriction, not
evidence that the host Nix installation is broken.
- Built `edge_infer_bench-0.1.1-py3-none-any.whl` and `edge_infer_bench-0.1.1.tar.gz`;
  the wheel was installed into the isolated CPU environment and its CLI/worker ran
  from `/tmp`. No package was published.

## v0.1.1 standard batch-1 measurements

A controlled standard-mode run was completed for the retained ResNet50 and
MobileNetV2 artifacts at batch 1. It used five independent child trials, 20 warmups,
200 timed iterations, four requested CPU threads, seed 42, and sequential backend
runs. These are actual measurements from the same shared AMD host; power, thermal,
and background load were not controlled. Full raw trials and reports are in
[`examples/measured/v011-standard`](../examples/measured/v011-standard).

| Model | Backend | P50 median (ms) | P95 median (ms) | P99 median (ms) | Throughput median (samples/s) | Peak RSS median (MB) | P50 bootstrap CI95 (ms) |
|---|---|---:|---:|---:|---:|---:|---|
| ResNet50 | PyTorch CPU | 156.998 | 190.610 | 220.968 | 6.300 | 828.457 | 145.399–179.675 |
| ResNet50 | ONNX Runtime CPU | 61.968 | 72.751 | 78.082 | 16.692 | 339.689 | 59.625–64.127 |
| MobileNetV2 | PyTorch CPU | 46.796 | 61.068 | 69.518 | 21.153 | 737.092 | 45.011–48.155 |
| MobileNetV2 | ONNX Runtime CPU | 8.117 | 11.177 | 12.121 | 121.996 | 125.194 | 7.925–8.514 |

These rows compare matching FP32 weights, input contract, batch, seed, thread
request, and method on one machine. They are still microbenchmarks, and the confidence
intervals are exploratory because they summarize only five independent trials.

The retained v0.1.1 smoke evidence is in
[`examples/measured/v011-smoke`](../examples/measured/v011-smoke). It uses the tiny
network, random exact weights, FP32 NCHW input, and three trials with five warmups
and 30 measurements. It demonstrates the real process protocol and report schema,
not a representative image-model performance claim.

| Backend | Model | Batch | Trials | P50 median (ms) | Throughput median (samples/s) | Peak RSS median (MB) |
|---|---|---:|---:|---:|---:|---:|
| ONNX Runtime CPU | tiny | 1 | 3 | 0.030 | 28,757.035 | 87.896 |
| OpenVINO CPU | tiny | 1 | 3 | 0.084 | 11,275.872 | 115.225 |

The numbers above are smoke measurements with 30 iterations and should not be
compared with the v0.1.0 ResNet/MobileNet short runs as a backend ranking.

## v0.1.0 baseline

The original development measurements remain documented verbatim in the retained
v0.1.0 report directory. They used CUDA-capable PyTorch wheels on CPU, one process
per backend, five warmups, 30 iterations, and no repeated-trial aggregation. They are
short development measurements, not authoritative performance conclusions. See
[`examples/measured/ryzen4800h`](../examples/measured/ryzen4800h/) and the v0.1.0
sections of the original artifacts for the full table and numerical validation.

## Remaining limitations

The full batch-size 1/4/8/16 ResNet50 and MobileNetV2 v0.1.1 matrix was not repeated;
the retained v0.1.0 measurements remain the comparison baseline for those additional batches. CPU frequency,
thermal state, background load, and power mode are best-effort metadata rather than
controlled variables. Process RSS is whole-process memory and sampled every 20 ms,
so observed peaks are lower bounds. Numerical equivalence covers tested tensors and
does not establish predictive accuracy. OpenVINO success was demonstrated for tiny,
not every torchvision model. See [reproducibility](reproducibility.md),
[methodology](benchmark-methodology.md), and [OpenVINO notes](openvino-compatibility.md).
