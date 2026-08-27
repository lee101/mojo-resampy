from __future__ import annotations

import inspect

import numpy as np
import pytest
import resampy

import mojo_resampy as mrs

RNG = np.random.default_rng(2026)


def assert_parity(actual, expected):
    if expected.dtype in (np.float32, np.complex64):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


def test_public_signatures_match_upstream():
    assert inspect.signature(mrs.resample) == inspect.signature(resampy.resample)
    assert inspect.signature(mrs.resample_nu) == inspect.signature(resampy.resample_nu)


@pytest.mark.parametrize("name", ["kaiser_fast", "kaiser_best"])
def test_precomputed_filter_matches_upstream(name):
    actual = mrs.filters.get_filter(name)
    expected = resampy.filters.get_filter(name)
    assert actual[1:] == expected[1:]
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("precision", [0, 5, 9])
@pytest.mark.parametrize("rolloff", [0.8, 0.945, 1.0])
def test_sinc_window_matches_upstream(precision, rolloff):
    actual = mrs.filters.sinc_window(
        num_zeros=7, precision=precision, rolloff=rolloff
    )
    expected = resampy.filters.sinc_window(
        num_zeros=7, precision=precision, rolloff=rolloff
    )
    assert actual[1:] == expected[1:]
    np.testing.assert_array_equal(actual[0], expected[0])


def test_callable_filter_matches_upstream():
    def triangular(num_zeros=5, precision=4):
        size = num_zeros * 2**precision + 1
        return np.linspace(1.0, 0.0, size), 2**precision, 1.0

    x = RNG.normal(size=257)
    assert_parity(
        mrs.resample(x, 100, 137, filter=triangular, num_zeros=4),
        resampy.resample(x, 100, 137, filter=triangular, num_zeros=4),
    )


def test_filter_cache_and_clear():
    mrs.filters.clear_cache()
    assert not mrs.filters.FILTER_CACHE
    first = mrs.filters.get_filter("kaiser_fast")
    assert mrs.filters.FILTER_CACHE
    assert mrs.filters.get_filter("kaiser_fast") is first
    mrs.filters.clear_cache()
    assert not mrs.filters.FILTER_CACHE


def test_missing_filter_raises():
    with pytest.raises(NotImplementedError):
        mrs.filters.get_filter("not-a-filter")


@pytest.mark.parametrize("filter_name", ["kaiser_fast", "kaiser_best", "sinc_window"])
@pytest.mark.parametrize("rates", [(48000, 44100), (22050, 48000), (100, 37)])
def test_resample_float64_parity(filter_name, rates):
    x = RNG.normal(size=503)
    assert_parity(
        mrs.resample(x, *rates, filter=filter_name),
        resampy.resample(x, *rates, filter=filter_name),
    )


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_resample_dtype_and_values(dtype):
    real = RNG.normal(size=211)
    if np.issubdtype(dtype, np.complexfloating):
        x = (real + 1j * RNG.normal(size=211)).astype(dtype)
    else:
        x = real.astype(dtype)
    actual = mrs.resample(x, 32000, 44100, filter="kaiser_fast")
    expected = resampy.resample(x, 32000, 44100, filter="kaiser_fast")
    assert actual.dtype == expected.dtype == x.dtype
    assert_parity(actual, expected)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_integer_input_promotes_and_matches(dtype):
    x = RNG.integers(-30000, 30001, size=241, dtype=dtype)
    actual = mrs.resample(x, 44100, 32000, filter="kaiser_fast")
    expected = resampy.resample(x, 44100, 32000, filter="kaiser_fast")
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-2)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_arbitrary_axis_shape_and_values(axis):
    x = RNG.normal(size=(5, 17, 7))
    actual = mrs.resample(x, 100, 61, axis=axis, filter="kaiser_fast")
    expected = resampy.resample(x, 100, 61, axis=axis, filter="kaiser_fast")
    assert actual.shape == expected.shape
    assert_parity(actual, expected)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("axis", [0, -1])
def test_contiguity_matches_upstream(order, axis):
    x = np.array(RNG.normal(size=(7, 31)), order=order)
    actual = mrs.resample(x, 10, 13, axis=axis, filter="kaiser_fast")
    expected = resampy.resample(x, 10, 13, axis=axis, filter="kaiser_fast")
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    assert_parity(actual, expected)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_nonuniform_resampling_parity(axis):
    x = RNG.normal(size=(9, 31))
    axis = axis if axis != 1 or x.ndim > 1 else -1
    end = (x.shape[axis] - 1) / 80.0
    times = np.sort(RNG.uniform(0.0, end, size=23))
    actual = mrs.resample_nu(
        x, 80.0, times, axis=axis, filter="kaiser_fast", parallel=True
    )
    expected = resampy.resample_nu(
        x, 80.0, times, axis=axis, filter="kaiser_fast", parallel=False
    )
    assert_parity(actual, expected)


def test_nonuniform_integer_dtype():
    x = RNG.integers(-2000, 2000, size=100, dtype=np.int32)
    times = np.arange(199) / 200.0
    actual = mrs.resample_nu(x, 100, times, filter="kaiser_fast")
    expected = resampy.resample_nu(x, 100, times, filter="kaiser_fast")
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-3)


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
def test_nonuniform_supported_dtypes(dtype):
    real = RNG.normal(size=101)
    if np.issubdtype(dtype, np.complexfloating):
        x = (real + 1j * RNG.normal(size=101)).astype(dtype)
    else:
        x = real.astype(dtype)
    times = np.linspace(0.0, 1.0, 73)
    actual = mrs.resample_nu(x, 100, times, filter="kaiser_fast")
    expected = resampy.resample_nu(x, 100, times, filter="kaiser_fast")
    assert actual.dtype == expected.dtype == x.dtype
    assert_parity(actual, expected)


def test_equal_rates_return_owned_copy():
    x = RNG.normal(size=100)
    y = mrs.resample(x, 44100, 44100)
    np.testing.assert_array_equal(y, x)
    assert y is not x
    assert y.flags.owndata


def test_length_rounding_regression():
    x = np.zeros(12499)
    assert mrs.resample(x, 12499, 15001).shape == (15001,)


@pytest.mark.parametrize("sr_orig,sr_new", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_invalid_sample_rates(sr_orig, sr_new):
    with pytest.raises(ValueError):
        mrs.resample(np.zeros(20), sr_orig, sr_new)


@pytest.mark.parametrize("sr_orig", [0, -1])
def test_invalid_nonuniform_sample_rate(sr_orig):
    with pytest.raises(ValueError):
        mrs.resample_nu(np.zeros(20), sr_orig, np.array([0.0]))


@pytest.mark.parametrize("times", [np.empty(0), np.eye(2)])
def test_invalid_nonuniform_times(times):
    with pytest.raises(ValueError):
        mrs.resample_nu(np.zeros(20), 1, times)


@pytest.mark.parametrize("times", [np.array([-0.1, 0.0]), np.array([0.0, 20.0])])
def test_nonuniform_domain_check(times):
    with pytest.raises(ValueError):
        mrs.resample_nu(np.zeros(20), 1, times)


def test_short_signal_rejected():
    with pytest.raises(ValueError):
        mrs.resample(np.zeros(2), 4, 1)


def test_empty_nonuniform_input_is_rejected_before_ffi():
    with pytest.raises(ValueError):
        mrs.resample_nu(np.empty(0), 1, np.array([0.0]))


@pytest.mark.parametrize("times", [np.array([np.nan]), np.array([np.inf])])
def test_nonfinite_nonuniform_times_are_rejected_before_ffi(times):
    with pytest.raises(ValueError, match="finite"):
        mrs.resample_nu(np.ones(4), 1, times)


@pytest.mark.parametrize(
    "definition,error",
    [
        ((np.array([]), 1, 1.0), ValueError),
        ((np.ones(4), 0, 1.0), ValueError),
        ((np.ones(4), 1.5, 1.0), TypeError),
        ((np.array([1.0, np.nan]), 1, 1.0), ValueError),
        ((np.array([1.0 + 1.0j, 0.0]), 1, 1.0), TypeError),
    ],
)
def test_malformed_custom_filter_is_rejected_before_ffi(definition, error):
    def malformed_filter():
        return definition

    with pytest.raises(error):
        mrs.resample(np.ones(8), 1, 2, filter=malformed_filter)


def test_unsupported_dtype_rejected_even_at_equal_rates():
    with pytest.raises(TypeError):
        mrs.resample(np.ones(8, dtype=np.float16), 1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rolloff": 0.0},
        {"rolloff": 1.1},
        {"num_zeros": 0},
        {"precision": -1},
        {"window": np.ones(10)},
    ],
)
def test_invalid_sinc_filter_arguments(kwargs):
    error = TypeError if "window" in kwargs else ValueError
    with pytest.raises(error):
        mrs.filters.sinc_window(**kwargs)


def test_parallel_flag_has_identical_numerics():
    x = RNG.normal(size=500)
    sequential = mrs.resample(x, 44100, 48000, parallel=False)
    parallel = mrs.resample(x, 44100, 48000, parallel=True)
    np.testing.assert_array_equal(sequential, parallel)


def test_packed_simd_tail_matches_upstream():
    def eleven_tap_filter():
        precision = 16
        size = 11 * precision + 1
        return np.linspace(1.0, 0.0, size), precision, 1.0

    x = RNG.normal(size=2048)
    times = np.linspace(0.0, x.size - 1, 1501)
    actual = mrs.resample_nu(x, 1.0, times, filter=eleven_tap_filter)
    expected = resampy.resample_nu(x, 1.0, times, filter=eleven_tap_filter)
    assert_parity(actual, expected)


@pytest.mark.parametrize("n_output", [32767, 32768, 32771])
def test_parallel_threshold_matches_serial(n_output):
    def short_filter():
        precision = 4
        return np.linspace(1.0, 0.0, 3 * precision + 1), precision, 1.0

    x = RNG.normal(size=1024)
    times = np.linspace(0.0, x.size - 1, n_output)
    sequential = mrs.resample_nu(
        x, 1.0, times, filter=short_filter, parallel=False
    )
    threaded = mrs.resample_nu(
        x, 1.0, times, filter=short_filter, parallel=True
    )
    np.testing.assert_array_equal(sequential, threaded)


def test_downsample_packed_simd_tail_matches_upstream():
    def seventeen_tap_filter():
        precision = 16
        size = 17 * precision + 1
        return np.linspace(1.0, 0.0, size), precision, 1.0

    x = RNG.normal(size=50000)
    actual = mrs.resample(
        x, 50000, 32771, filter=seventeen_tap_filter, parallel=True
    )
    expected = resampy.resample(
        x, 50000, 32771, filter=seventeen_tap_filter, parallel=False
    )
    assert_parity(actual, expected)


def test_passband_tone_quality():
    sr_orig, sr_new, frequency = 44100, 22050, 512
    t_in = np.arange(2 * sr_orig) / sr_orig
    t_expected = np.arange(2 * sr_new) / sr_new
    x = np.sin(2 * np.pi * frequency * t_in)
    expected = np.sin(2 * np.pi * frequency * t_expected)
    actual = mrs.resample(x, sr_orig, sr_new)
    interior = slice(sr_new // 2, -sr_new // 2)
    assert np.mean(np.abs(actual[interior] - expected[interior])) < 1e-7
