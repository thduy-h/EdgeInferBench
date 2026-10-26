# Benchmark methodology

## What is measured

Each backend runs synchronous FP32 CPU inference on the same deterministic,
contiguous NCHW standard-normal tensor. A row corresponds to one batch size. Input
generation and backend-specific tensor preparation occur before measurement. Timing
uses `time.perf_counter_ns()` immediately around the adapter's `infer` call, including
output normalization to host NumPy tensors. PyTorch uses inference mode; ORT selects
`CPUExecutionProvider`; OpenVINO explicitly compiles for CPU and uses a synchronous
infer request. No asynchronous queueing or inferred completion times are involved.

Library imports, model loading, state restoration, graph checking, ORT session
optimization, OpenVINO compilation, and report serialization are excluded from
steady-state latency. `initialization_ms` records adapter setup after runtime import;
it is useful diagnostic context, not a standardized cross-backend startup benchmark.
`first_inference_ms` records an additional untimed-for-steady-state first execution
at each shape. It may include lazy shape-specific compilation/allocation. Backends
can still have lazy work later; inspect raw timings rather than assuming warmup
guarantees perfect stability.

## Trial and process isolation

`edgebench benchmark` launches a fresh `python -m edge_infer_bench.benchmark_worker`
interpreter for every trial. The request and result use temporary JSON files, while
stdout and stderr are captured separately. The child imports only the requested
backend, prepares deterministic inputs, loads and initializes the runtime, warms up,
measures, samples RSS, writes one structured trial result, and exits. A timeout,
crash, missing result, or invalid JSON is retained as a failed trial with
diagnostics; failed trials are never silently discarded.

The standard CLI mode uses five trials, 20 warmups, and 200 measurements. Smoke mode
uses three trials, five warmups, and 30 measurements. The Python API keeps
`BenchmarkConfig(trials=1)` as its backwards-compatible default. JSON retains the
raw `trials` array and exact `execution_order`.

## Warmup and independent batches

Every batch gets one first inference followed by the configured number of warmup
iterations, then the configured number of measurements. A model/session is reused
across batch rows in a run, but every row receives a new deterministic input and its
own warmup. No observations are silently dropped. A child is reused for the configured batch
sizes within one trial; invoke the benchmark separately for each batch when strict
per-batch memory isolation is required. Warmup counts of zero are allowed for
testing; use longer warmup and measurements for engineering decisions.

## Latency percentiles and throughput

Nanosecond durations are stored in JSON. Millisecond P50/P95/P99 use
`numpy.percentile` with its default linear interpolation. Mean, min, max, and
population standard deviation (`ddof=0`) are computed from all timed samples.
Short runs provide particularly weak tail estimates; 200 iterations is a starting
point, not a statistical guarantee. Repeat independent trials and report variability.

Aggregated P50/P95/P99 and throughput are medians of trial-level summaries.
Between-trial standard deviations are calculated across completed trials. With at
least three completed trials, JSON includes a percentile-bootstrap 95% interval over
independent trial summaries (4,000 resamples). This is exploratory with only a few
trials; correlated inference iterations are never treated as independent trials.
With fewer than three completed trials, the interval is explicitly `null`.

`throughput_samples_s = (batch_size * measurement_iterations) / sum(timed_seconds)`.
This is synchronous serial inference throughput for the measured scope, excluding
Python loop bookkeeping between calls. It is not reciprocal P50/P95, saturation
throughput of a concurrent service, or end-to-end application throughput. Actual
processed sample count and summed duration are recorded.

## Process memory

psutil measures RSS for the entire benchmark process, in decimal MB (1,000,000
bytes), including Python, native libraries, runtime graph structures, cached
allocations, inputs, and outputs. It is neither model-exclusive allocation nor
total system RAM. Artifact MB counts serialized model payloads (ONNX embedded weights,
IR XML+BIN, or reference state+metadata), not loaded-memory size. ONNX weights are
embedded to keep hashing and artifact sizing unambiguous in v0.1.

Baseline RSS is captured before adapter loading. A daemon samples RSS every 20 ms
during warmup and measurement, with initial/final observations outside the timing
loop. The recorded peak is the **peak observed** process RSS in that window, a lower
bound on the true peak; it excludes first inference and initialization. Sampling
does not add psutil calls to each timed inference but has a small scheduling cost.
RSS after initialization, peak observed RSS, incremental RSS relative to baseline,
and RSS after measurement are recorded separately. Allocators and earlier batches can
retain memory. Use separate processes/batches to interpret differences cautiously.
The end RSS read occurs after statistics bookkeeping; a small increase after the
sampler window can make this value exceed the window's peak observed RSS.

## Numerical correctness

Export saves the exact original state dictionary and metadata before conversion.
Restoration verifies its SHA-256 and uses `torch.load(weights_only=True)` with strict
state matching. Do not load untrusted checkpoints or graphs. Graph integrity uses
ONNX's full checker, not file existence. Every ONNX backend load checks the graph.
OpenVINO conversion goes through checked ONNX and saves with FP16 compression
disabled. An FP32 execution hint avoids implicit BF16-oriented CPU defaults.

Each test input uses a local NumPy generator seeded by `[seed, batch_size,
input_index]`; no shared RNG progression depends on case ordering. Both backends
receive the identical array. Output trees flatten lists/tuples and insertion-ordered
dictionaries into positional arrays. Output count, shape, dtype, emptiness, and
finite values are checked. Empty or non-finite outputs fail.

Absolute error is `abs(target-reference)`. Reported relative error divides by
`max(abs(reference), 1e-8)` to remain defined near zero. Max/mean metrics use float64
arithmetic. Pass/fail uses `numpy.allclose(target, reference, atol, rtol,
equal_nan=False)` plus equal dtype/shape/count, not a threshold on the maximum
relative metric. Thus very large relative error near zero need not indicate failure
when absolute error is within tolerance. Defaults are atol 1e-4 and rtol 1e-3;
change them deliberately and retain the reason in your experiment notes.

Conversion validates two inputs at each of batches 1 and 4 for dynamic models;
fixed models use their exported batch. Run `validate` at all benchmark batches to
extend this evidence. Numerical equivalence only covers tested inputs and does
not establish predictive accuracy or correctness for all possible inputs. Random
models are appropriate for performance/fidelity tests, not classification accuracy.

## CPU threads and fairness

`--threads` sets PyTorch intra-op threads, ORT intra-op threads, or OpenVINO
INFERENCE_NUM_THREADS. PyTorch requests one inter-op thread; if an embedding process
already started parallel work, PyTorch can reject changing inter-op threads and the
actual value is recorded. Use fresh CLI processes to avoid this case. ORT uses one
inter-op thread, sequential execution, full graph optimization, and disabled
intra-op spinning. OpenVINO uses LATENCY, one stream, and an FP32 inference hint.
Effective backend settings, versions, affinity, governor, and relevant thread
environment variables are recorded. These policies differ and can affect results.

Matching requested threads does not mean equal scheduling, vectorization, operator
fusion, or core utilization. Hardware temperature, frequency boost, background
load, power mode, and benchmark order are not controlled automatically. Do not run
benchmark processes concurrently. Test an idle machine, keep power conditions
stable, alternate backend run order, and repeat trials. Pin affinity externally
with `taskset` if desired and record that choice. No system configuration changes
are performed by EdgeInferBench.

## Reports and interpretation

Comparison groups require equal source checkpoint identity, input contract, FP32
precision, batch size, deterministic input SHA-256, seed, warmup, measurement count, requested threads, CPU,
architecture, OS, Python, affinity, governor, thread environment, and dependency
versions. Runtime-specific settings are shown for review rather than claimed to be
identical. Rows with mismatched contracts form separate groups. Charts have per-group
scales; use numeric values when interpreting batch scaling. CSV retains group IDs
and failure reasons; JSON retains raw runs and full validation detail. A missing
backend is skipped explicitly; a runtime/shape/numerical problem is never encoded
as zero latency or success.

Reports retain every trial and the aggregate; they do not silently average or rank
incompatible runs. HTML embeds correctness checks from conversion metadata
and explicitly supplied validation JSON. Matching groups support local comparisons,
not a universal backend ranking. A conversion's built-in checks do not imply that
all benchmark shapes have been validated; retain explicit all-batch validation JSON.

Microbenchmarks are not equivalent to end-to-end application performance: real
services also include decoding, preprocessing, I/O, batching queues, concurrency,
postprocessing, and startup costs. Interpret latency, throughput, memory, and
numerical fidelity together. Preserve model bundles, sidecars, raw JSON, dependency
locks, source version, and exact commands to reproduce an experiment.
