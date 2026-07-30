"""Benchmark mojo-resampy against upstream resampy on identical arrays."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import resampy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_resampy as mrs  # noqa: E402


def timeit(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf8") as file:
            for line in file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    rng = np.random.default_rng(7)
    cases = []

    x_best = np.ascontiguousarray(rng.normal(size=480_000))
    x_fast = np.ascontiguousarray(rng.normal(size=441_000))
    stereo = np.ascontiguousarray(rng.normal(size=(2, 240_000)))
    x_nu = np.ascontiguousarray(rng.normal(size=240_000))
    t_out = np.sort(rng.uniform(0, (x_nu.size - 1) / 48000, size=180_000))

    def add_cases(parallel: bool, mode: str) -> None:
        cases.extend(
            [
                (
                    f"resample 10 s mono 48k->44.1k, best, {mode}",
                    lambda: mrs.resample(
                        x_best,
                        48000,
                        44100,
                        filter="kaiser_best",
                        parallel=parallel,
                    ),
                    lambda: resampy.resample(
                        x_best,
                        48000,
                        44100,
                        filter="kaiser_best",
                        parallel=parallel,
                    ),
                ),
                (
                    f"resample 10 s mono 44.1k->48k, fast, {mode}",
                    lambda: mrs.resample(
                        x_fast,
                        44100,
                        48000,
                        filter="kaiser_fast",
                        parallel=parallel,
                    ),
                    lambda: resampy.resample(
                        x_fast,
                        44100,
                        48000,
                        filter="kaiser_fast",
                        parallel=parallel,
                    ),
                ),
                (
                    f"resample 5 s stereo 48k->16k, fast, {mode}",
                    lambda: mrs.resample(
                        stereo,
                        48000,
                        16000,
                        axis=1,
                        filter="kaiser_fast",
                        parallel=parallel,
                    ),
                    lambda: resampy.resample(
                        stereo,
                        48000,
                        16000,
                        axis=1,
                        filter="kaiser_fast",
                        parallel=parallel,
                    ),
                ),
                (
                    f"resample_nu 240k->180k, fast, {mode}",
                    lambda: mrs.resample_nu(
                        x_nu,
                        48000,
                        t_out,
                        filter="kaiser_fast",
                        parallel=parallel,
                    ),
                    lambda: resampy.resample_nu(
                        x_nu,
                        48000,
                        t_out,
                        filter="kaiser_fast",
                        parallel=parallel,
                    ),
                ),
            ]
        )

    add_cases(False, "serial")
    add_cases(True, "parallel")

    print(f"Machine: {cpu_name()}")
    print(f"Python: {platform.python_version()}, resampy: {resampy.__version__}")
    print()
    print("| case | mojo-resampy | resampy | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases:
        ours()
        upstream()
        mojo_seconds = timeit(ours)
        upstream_seconds = timeit(upstream)
        ratio = upstream_seconds / mojo_seconds
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
