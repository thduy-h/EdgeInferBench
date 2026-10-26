# Actual Ryzen 7 4800H development trial

See [execution notes and all numeric results](../../../docs/development-results.md).

- `benchmark.html`: self-contained offline report, generated from retained raw results.
- `benchmark.csv`: flattened measurements and explicit validation records.
- `benchmark.json`: aggregate of unmodified recorded benchmark/validation JSON.
- `measurements/`: four real benchmark runs, two all-batch correctness validations,
  and one OpenVINO availability trial whose four measurements are skipped.
- `environment.json`: actual `edgebench info --json` output.
- `environment-freeze.txt`: actual package snapshot, including the reused CUDA-capable
  torch build's native dependencies. Inference was exclusively CPU. This is not a
  portable dependency lock or a CUDA installation recommendation.
- `openvino-conversion.txt`: captured failed conversion diagnostic; no IR was created.
- `verification.txt`: executed test, lint, build, and installation verification notes.

FP32, seed 42, random model weights, batches 1/4/8/16, 4 requested threads,
5 warmups and 30 measurements per batch. These are short trials on a shared
development machine; do not infer universal rankings or reliable tail estimates.
Timestamps in JSON are UTC (26 October); the local date was 27 October 2026.
