"""Validated, backend-independent experiment settings."""

import math
from dataclasses import dataclass


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "Batch sizes must be comma-separated positive integers, e.g. 1,4,8"
        ) from exc
    if not sizes or any(size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Batch sizes must be positive and unique")
    return sizes


@dataclass(frozen=True)
class InputSpec:
    channels: int = 3
    height: int = 224
    width: int = 224
    dtype: str = "float32"
    layout: str = "NCHW"
    distribution: str = "standard_normal"

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.channels, self.height, self.width)):
            raise ValueError("Input dimensions must be positive")
        if (self.dtype, self.layout, self.distribution) != ("float32", "NCHW", "standard_normal"):
            raise ValueError("v0.1 supports float32 NCHW standard_normal inputs")

    def shape(self, batch: int) -> tuple[int, int, int, int]:
        if batch < 1:
            raise ValueError("Batch size must be positive")
        return batch, self.channels, self.height, self.width


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iterations: int = 20
    measurement_iterations: int = 200
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16)
    cpu_threads: int = 4
    random_seed: int = 42
    trials: int = 1
    timeout_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if any(
            type(v) is not int
            for v in (
                self.warmup_iterations,
                self.measurement_iterations,
                self.cpu_threads,
                self.random_seed,
            )
        ):
            raise ValueError("Iterations, threads, and seed must be integers")
        if self.warmup_iterations < 0 or self.measurement_iterations < 1:
            raise ValueError("Warmup must be >= 0 and measurement iterations >= 1")
        if self.cpu_threads < 1:
            raise ValueError("CPU threads must be >= 1")
        if self.trials < 1:
            raise ValueError("Trials must be >= 1")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Timeout must be finite and positive")
        parse_batch_sizes(",".join(map(str, self.batch_sizes)))
        if not 0 <= self.random_seed < 2**32:
            raise ValueError("Seed must be in [0, 2**32)")


def check_tolerances(atol: float, rtol: float) -> None:
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError("atol and rtol must be finite and non-negative")
