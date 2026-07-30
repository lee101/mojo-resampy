# mojo-resampy

`mojo-resampy` is a standalone Mojo implementation of
[resampy](https://github.com/bmcfee/resampy)'s compute-bound, band-limited
sample-rate conversion loop. Its Python API follows resampy 0.4.3 for the
documented subset below.

Use it by changing one import:

```python
import numpy as np
import mojo_resampy as resampy

sample_rate = 48_000
t = np.arange(sample_rate, dtype=np.float64) / sample_rate
x = np.sin(2 * np.pi * 440 * t)

y = resampy.resample(x, sample_rate, 16_000, filter="kaiser_best")
print(y.shape)  # (16000,)
```

## Coverage

The public callables of upstream resampy 0.4.3 are present, with this coverage:

| API | Coverage |
| --- | --- |
| `resample(x, sr_orig, sr_new, axis=-1, filter="kaiser_best", parallel=False, **kwargs)` | Uniform resampling, arbitrary axes and dimensions |
| `resample_nu(x, sr_orig, t_out, axis=-1, filter="kaiser_best", parallel=False, **kwargs)` | Interpolation at arbitrary output times |
| `filters.get_filter`, `filters.sinc_window`, `filters.clear_cache` | Built-in, named, and callable custom filters |
| `kaiser_best`, `kaiser_fast` | Generated from the same tuned parameters as upstream |
| Numeric types | `float32`, `float64`, `complex64`, `complex128`; integer inputs promote to `float32` |
| Layout | C- and Fortran-contiguous inputs, with upstream-compatible output layout |

The `parallel` keyword selects thresholded CPU execution for float64 and
complex128 kernels. Smaller calls stay serial and larger calls are divided into
independent output chunks. Other supported dtypes accept the keyword but
currently stay serial. The package does not expose resampy's private Numba
generalized ufuncs. Boolean and float16 input, streaming/stateful resampling,
GPU execution, and a native Mojo filter-design API are not covered.

The tests compare every covered behavior directly against the real `resampy`
package, including both rate directions, all supported dtypes, complex data,
multiple axes, non-uniform times, filter coefficients, validation failures,
contiguity, and passband quality.

## Install

From this repository:

```bash
pixi install
pixi run build
pixi run test
pixi run python examples/basic.py
```

`pixi install` creates the pinned Python and Mojo environment. `pixi run build`
compiles `src/resample.mojo` into `dist/libmojo-resampy.so`. Imports also rebuild
the library when the Mojo source is newer. Set `MOJO_RESAMPY_LIB` to use an
already-built shared library at another path.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, using
Python 3.13.14 and resampy 0.4.3. Each row reports the best of three warm runs
on identical contiguous float64 arrays. Serial rows use `parallel=False` for
both libraries; parallel rows use `parallel=True` for both.

| case | mojo-resampy | resampy | ratio |
| --- | ---: | ---: | ---: |
| resample 10 s mono 48k->44.1k, best, serial | 223.47 ms | 303.43 ms | 1.36x faster |
| resample 10 s mono 44.1k->48k, fast, serial | 33.79 ms | 82.75 ms | 2.45x faster |
| resample 5 s stereo 48k->16k, fast, serial | 42.89 ms | 65.27 ms | 1.52x faster |
| resample_nu 240k->180k, fast, serial | 12.66 ms | 29.41 ms | 2.32x faster |
| resample 10 s mono 48k->44.1k, best, parallel | 41.68 ms | 36.62 ms | 0.88x slower |
| resample 10 s mono 44.1k->48k, fast, parallel | 14.77 ms | 11.45 ms | 0.78x slower |
| resample 5 s stereo 48k->16k, fast, parallel | 9.50 ms | 16.86 ms | 1.77x faster |
| resample_nu 240k->180k, fast, parallel | 7.92 ms | 5.33 ms | 0.67x slower |

These are measurements from one machine, not general performance guarantees.
Run `pixi run bench` to reproduce the table on your hardware.

## How it works

Python performs API validation, filter construction, axis normalization, and
allocation. It moves the resampled axis to a contiguous trailing dimension and
makes one ctypes call for the full operation. Real and complex arrays use
dedicated float32 or float64 Mojo kernels; complex samples cross as interleaved
real and imaginary components. Integer data crosses through a float64 work
buffer and accumulates to the upstream-compatible float32 output.

The FFI owns no memory. NumPy owns the input, output, time, and filter arrays,
and passes their addresses as 64-bit integers. The exported Mojo functions
reconstruct `UnsafePointer[..., AnyOrigin[mut=True]]` values inside the C ABI
boundary and write into caller-provided row-major buffers.

For each output position, Mojo splits the symmetric low-pass filter into left
and right wings. It indexes the oversampled half-window, linearly interpolates
between adjacent coefficients, clips each wing at the signal boundary, and
accumulates the weighted input samples. Downsampling scales both the filter and
its table stride to suppress frequencies above the new Nyquist limit.

For scale-1 float64 kernels, Python packs each filter phase into contiguous tap
order. Mojo processes those taps with the native float64 SIMD width, two vector
accumulators, and an explicit scalar tail. Packing is enabled only when there
are at least 1,024 output points; short calls and downsampling avoid its
overhead. Parallel calls above the threshold dispatch independent output
chunks while capturing only raw buffer addresses across the worker boundary.

## License

MIT
