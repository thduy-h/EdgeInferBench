# Changelog

## 0.1.1 - 2026-10-27

Maintenance release focused on reproducibility and release hardening.

- Run every benchmark trial in a fresh child Python interpreter.
- Retain trial P50/P95/P99, throughput, RSS, process IDs, input checksums, and diagnostics.
- Aggregate independent trials with medians, between-trial variability, and exploratory bootstrap intervals.
- Record baseline, post-initialization, peak observed, incremental, and post-benchmark RSS.
- Preserve explicit timeout, crash, skipped-backend, partial-trial, and missing-result states.
- Add failure-injection and subprocess-isolation integration coverage.
- Accept and render v0.1.0 and v0.1.1 benchmark JSON schemas.
- Document CPU-only installation, reproducibility, NixOS checks, and OpenVINO status.
- Keep OpenVINO and CUDA optional; core installation remains CPU-capable without either.

## 0.1.0 - 2026-10-27

Initial public toolkit release with PyTorch export, ONNX checking, optional OpenVINO
conversion, numerical validation, CPU backend adapters, benchmarking, reports, tests,
and documentation.
