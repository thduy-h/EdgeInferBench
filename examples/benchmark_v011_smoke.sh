#!/usr/bin/env bash
set -euo pipefail

# Versioned smoke example: three isolated trials, 5 warmups, 30 measurements.
# Set EDGE_ROOT to a new directory before running; existing artifacts are never overwritten.
root="${EDGE_ROOT:-examples/measured/v011-smoke-run}"
model="${MODEL:-tiny}"
mkdir -p "$root"
MODE=smoke TRIALS=3 WARMUP=5 ITERATIONS=30 \
  bash "$(dirname "$0")/benchmark_cpu.sh" "$root" "$model"
