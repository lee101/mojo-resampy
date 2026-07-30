"""Minimal mojo-resampy example."""

import numpy as np

import mojo_resampy as resampy


sample_rate = 48_000
t = np.arange(sample_rate, dtype=np.float64) / sample_rate
x = np.sin(2 * np.pi * 440 * t)

y = resampy.resample(x, sample_rate, 16_000, filter="kaiser_best")
print(y.shape)
