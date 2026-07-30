"""Public sample-rate conversion interface."""

from __future__ import annotations

import numpy as np

from . import _lib
from .filters import get_filter

__all__ = ["resample", "resample_nu"]


def _axis_index(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise np.exceptions.AxisError(axis, ndim=ndim)
    return axis


def _output_dtype(dtype: np.dtype) -> np.dtype:
    if np.issubdtype(dtype, np.integer):
        return np.dtype(np.float32)
    if dtype in (np.float32, np.float64, np.complex64, np.complex128):
        return dtype
    raise TypeError(f"Unsupported input dtype: {dtype}")


def _apply(
    x: np.ndarray,
    times: np.ndarray,
    axis: int,
    interp_win: np.ndarray,
    precision: int,
    scale: float,
    time_scale: float,
    window_scale: float,
    parallel: bool,
) -> np.ndarray:
    axis = _axis_index(axis, x.ndim)
    n_input = x.shape[axis]
    n_output = times.size
    output_dtype = _output_dtype(x.dtype)
    if n_input < 1 or n_output < 1:
        raise ValueError("input and output buffers must be non-empty")
    if not isinstance(precision, (int, np.integer)) or isinstance(
        precision, (bool, np.bool_)
    ):
        raise TypeError("filter precision must be an integer")
    precision = int(precision)
    if precision < 1:
        raise ValueError("filter precision must be positive")
    if not np.isfinite((scale, time_scale, window_scale)).all():
        raise ValueError("resampling scales must be finite")
    real_dtype = np.float32 if output_dtype in (np.float32, np.complex64) else np.float64
    components = 2 if np.issubdtype(output_dtype, np.complexfloating) else 1
    integer_input = np.issubdtype(x.dtype, np.integer)

    moved = np.moveaxis(x, axis, -1)
    work_dtype = np.float64 if integer_input else output_dtype
    work = np.ascontiguousarray(moved, dtype=work_dtype)
    output_shape = moved.shape[:-1] + (n_output,)
    result_moved = np.empty(output_shape, dtype=output_dtype, order="C")

    input_real_dtype = np.float64 if integer_input else real_dtype
    real_input = work.view(input_real_dtype)
    real_output = result_moved.view(real_dtype)
    if np.iscomplexobj(times):
        raise TypeError("output times must be real-valued")
    times = np.ascontiguousarray(times, dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("output times must be finite")
    if np.iscomplexobj(interp_win):
        raise TypeError("filter coefficients must be real-valued")
    interp_win = np.ascontiguousarray(interp_win, dtype=np.float64)
    n_window = interp_win.size
    if interp_win.ndim != 1 or n_window < 2:
        raise ValueError("filter coefficients must be a one-dimensional array of length >= 2")
    if not np.isfinite(interp_win).all():
        raise ValueError("filter coefficients must be finite")
    interp_delta = np.empty_like(interp_win)
    np.subtract(interp_win[1:], interp_win[:-1], out=interp_delta[:-1])
    interp_delta[-1] = 0.0
    n_series = work.size // n_input
    phase_taps = 0
    if (
        real_dtype == np.float64
        and scale == 1.0
        and n_series * n_output >= 1024
        and precision > 0
        and (n_window - 1) % precision == 0
    ):
        phase_taps = (n_window - 1) // precision
        packed_coefficients = np.empty(
            (2, precision, phase_taps), dtype=np.float64
        )
        packed_coefficients[0] = interp_win[:-1].reshape(
            phase_taps, precision
        ).T
        packed_coefficients[1] = interp_delta[:-1].reshape(
            phase_taps, precision
        ).T
        interp_win = packed_coefficients[0]
        interp_delta = packed_coefficients[1]

    if integer_input:
        fn = _lib.lib().mrs_resample_f64_to_f32
    elif real_dtype == np.float32:
        fn = _lib.lib().mrs_resample_f32
    else:
        fn = _lib.lib().mrs_resample_f64
    fn(
        _lib.addr(real_input),
        _lib.addr(times),
        _lib.addr(interp_win),
        _lib.addr(interp_delta),
        _lib.addr(real_output),
        n_series,
        n_input,
        n_output,
        n_window,
        int(precision),
        float(scale),
        float(time_scale),
        float(window_scale),
        phase_taps,
        components,
        int(parallel),
    )

    result = np.moveaxis(result_moved, -1, axis)
    if axis == x.ndim - 1 and not (x.flags.f_contiguous and not x.flags.c_contiguous):
        return result
    order = "F" if x.flags.f_contiguous and not x.flags.c_contiguous else "C"
    return np.array(result, dtype=output_dtype, order=order, copy=True)


def resample(
    x,
    sr_orig,
    sr_new,
    axis=-1,
    filter="kaiser_best",
    parallel=False,
    **kwargs,
):
    x = np.asarray(x)
    axis = _axis_index(axis, x.ndim)
    _output_dtype(x.dtype)
    if sr_orig <= 0:
        raise ValueError(f"Invalid sample rate: sr_orig={sr_orig}")
    if sr_new <= 0:
        raise ValueError(f"Invalid sample rate: sr_new={sr_new}")
    if sr_orig == sr_new:
        return x.copy()

    n_output = int(x.shape[axis] * float(sr_new) / float(sr_orig))
    if n_output < 1:
        raise ValueError(
            f"Input signal length={x.shape[axis]} is too small to "
            f"resample from {sr_orig}->{sr_new}"
        )

    sample_ratio = float(sr_new) / sr_orig
    interp_win, precision, _ = get_filter(filter, **kwargs)
    interp_win = np.asarray(interp_win)
    times = np.arange(n_output, dtype=np.float64) / sample_ratio
    return _apply(
        x,
        times,
        axis,
        interp_win,
        precision,
        min(1.0, sample_ratio),
        1.0,
        min(1.0, sample_ratio),
        bool(parallel),
    )


def resample_nu(
    x,
    sr_orig,
    t_out,
    axis=-1,
    filter="kaiser_best",
    parallel=False,
    **kwargs,
):
    x = np.asarray(x)
    axis = _axis_index(axis, x.ndim)
    _output_dtype(x.dtype)
    if sr_orig <= 0:
        raise ValueError(f"Invalid sample rate: sr_orig={sr_orig}")

    t_out = np.asarray(t_out)
    if t_out.ndim != 1:
        raise ValueError(f"Invalid t_out shape ({t_out.shape}), 1D array expected")
    if t_out.size == 0:
        raise ValueError("zero-size array to reduction operation minimum which has no identity")
    if np.iscomplexobj(t_out):
        raise TypeError("output times must be real-valued")
    if not np.isfinite(t_out).all():
        raise ValueError("output times must be finite")
    domain_end = (x.shape[axis] - 1) / sr_orig
    if np.min(t_out) < 0 or np.max(t_out) > domain_end:
        raise ValueError(
            f"Output domain [{np.min(t_out)}, {np.max(t_out)}] exceeds "
            f"the data domain [0, {domain_end}]"
        )

    interp_win, precision, _ = get_filter(filter, **kwargs)
    times = np.ascontiguousarray(t_out, dtype=np.float64)
    return _apply(
        x,
        times,
        axis,
        np.asarray(interp_win),
        precision,
        1.0,
        float(sr_orig),
        1.0,
        bool(parallel),
    )
