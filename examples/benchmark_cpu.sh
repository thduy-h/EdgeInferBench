#!/usr/bin/env bash
set -euo pipefail

# Run in an activated installation. Use a fresh output root for each trial.
run_root="${1:-experiment}"
model_name="${2:-mobilenet_v2}"
warmup_count="${WARMUP:-20}"
measurement_count="${ITERATIONS:-200}"
trial_count="${TRIALS:-5}"
benchmark_mode="${MODE:-standard}"
mkdir -p "$run_root/artifacts" "$run_root/results" "$run_root/reports"
edgebench info --json > "$run_root/environment.json"
edgebench export --model "$model_name" --output "$run_root/artifacts/$model_name.onnx" \
  --dynamic-batch --seed 42 --threads 4
edgebench validate --reference "$run_root/artifacts/${model_name}_reference" \
  --target "$run_root/artifacts/$model_name.onnx" --backend onnxruntime \
  --batch-sizes 1,4,8,16 --inputs 3 --threads 4 \
  --output "$run_root/results/${model_name}_validation.json"
edgebench benchmark --model "$run_root/artifacts/${model_name}_reference" --backend pytorch \
  --mode "$benchmark_mode" --trials "$trial_count" --batch-sizes 1,4,8,16 \
  --warmup "$warmup_count" --iterations "$measurement_count" --threads 4 \
  --output "$run_root/results/${model_name}_pytorch.json"
edgebench benchmark --model "$run_root/artifacts/$model_name.onnx" --backend onnxruntime \
  --mode "$benchmark_mode" --trials "$trial_count" --batch-sizes 1,4,8,16 \
  --warmup "$warmup_count" --iterations "$measurement_count" --threads 4 \
  --output "$run_root/results/${model_name}_onnxruntime.json"
for report_format in json csv html; do
  edgebench report --input "$run_root/results" --format "$report_format" \
    --output "$run_root/reports/benchmark.$report_format"
done
