# Reproducibility guide

EdgeInferBench separates deterministic inputs and model provenance from hardware
performance. A reproducible experiment keeps the reference bundle, converted model
and sidecars, validation JSON, every benchmark JSON, environment JSON, package
versions, and the exact command line together.

## Recommended run

Use a clean virtual environment and the CPU wheels documented in the README. Export
once with a fixed seed, validate every batch that will be benchmarked, then benchmark
with standard mode:

```bash
edgebench export --model resnet18 --format onnx --dynamic-batch --seed 42 \
  --output artifacts/resnet18.onnx
edgebench validate --reference artifacts/resnet18_reference \
  --target artifacts/resnet18.onnx --backend onnxruntime \
  --batch-sizes 1,4,8,16 --seed 42 --output results/resnet18-validation.json
edgebench benchmark --model artifacts/resnet18.onnx --backend onnxruntime \
  --mode standard --threads 4 --seed 42 --output results/resnet18-ort.json
```

Standard mode means five independent child processes, 20 warmups, and 200 timed
iterations per configured batch. Use `--mode smoke` for three trials, five warmups,
and 30 iterations while developing. Set `--trials`, `--warmup`, and `--iterations`
explicitly in automation so the command remains self-documenting.

Every child records its PID, executable, input SHA-256, backend settings, trial
status, and duration. The parent records execution order and keeps failed trials;
there is no zero-filled replacement for a missing measurement. A timeout is controlled
by `--timeout` seconds per child. Run backends sequentially or rotate their order
between repeated experiments to reduce thermal and background-load bias.

## Comparing machines

Compare only rows with the same source checksum, input shape and dtype, batch size,
seed, thread request, warmup/iteration counts, backend settings, and model precision.
Review CPU affinity, governor, frequency, thermal readings, kernel, Python and package
versions before interpreting a difference. Do not alter governors or require root.
CPU frequency and thermal fields are best-effort and may be unavailable.

Confidence intervals in schema 2 are percentile-bootstrap intervals over independent
trial summaries. They are omitted for fewer than three completed trials and are
exploratory with the default five. They are not confidence intervals over the 200
correlated inference calls inside one process.

Process RSS includes the interpreter, native runtime, loaded graph, allocator caches,
inputs, and outputs. Baseline, post-initialization, sampled peak, incremental, and
post-benchmark RSS are recorded; these values are not exact model memory usage. The
20 ms sampler observes a lower bound on very short peaks.

Numerical equivalence means the tested outputs match the exact exported weights under
the configured tolerances. It does not establish predictive accuracy, dataset
accuracy, or end-to-end application performance.
