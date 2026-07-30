"""Filter construction and loading compatible with :mod:`resampy.filters`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

FILTER_FUNCTIONS = ["sinc_window"]
FILTER_CACHE: dict[str, tuple[np.ndarray, int, float]] = {}

__all__ = ["get_filter", "clear_cache", "sinc_window"]

_PRECOMPUTED = {
    "kaiser_best": (50, 13, 12.984585247040016, 0.9173473712608761),
    "kaiser_fast": (24, 9, 9.903224416455274, 0.8682120388377784),
}


def sinc_window(
    num_zeros: int = 64,
    precision: int = 9,
    window: Callable[[int], np.ndarray] | None = None,
    rolloff: float = 0.945,
) -> tuple[np.ndarray, int, float]:
    if window is None:
        window = np.hanning
    elif not callable(window):
        raise TypeError(f"window must be callable, not type(window)={type(window)}")
    if not 0 < rolloff <= 1:
        raise ValueError(f"Invalid roll-off: rolloff={rolloff}")
    if num_zeros < 1:
        raise ValueError(f"Invalid num_zeros: num_zeros={num_zeros}")
    if precision < 0:
        raise ValueError(f"Invalid precision: precision={precision}")

    num_bits = 2**precision
    n = num_bits * num_zeros
    grid = np.linspace(0, num_zeros, num=n + 1, endpoint=True)
    sinc_win = rolloff * np.sinc(rolloff * grid)
    taper = np.asarray(window(2 * n + 1))[n:]
    return np.asarray(taper * sinc_win, dtype=np.float64), num_bits, rolloff


def load_filter(filter_name: str) -> tuple[np.ndarray, int, float]:
    if filter_name not in _PRECOMPUTED:
        raise OSError(f"unknown filter: {filter_name}")
    if filter_name not in FILTER_CACHE:
        num_zeros, precision, beta, rolloff = _PRECOMPUTED[filter_name]

        def kaiser(length: int) -> np.ndarray:
            return np.kaiser(length, beta)

        FILTER_CACHE[filter_name] = sinc_window(
            num_zeros=num_zeros,
            precision=precision,
            window=kaiser,
            rolloff=rolloff,
        )
    return FILTER_CACHE[filter_name]


def get_filter(name_or_function, **kwargs) -> tuple[np.ndarray, int, float]:
    if name_or_function == "sinc_window":
        return sinc_window(**kwargs)
    if callable(name_or_function):
        return name_or_function(**kwargs)
    try:
        return load_filter(name_or_function)
    except (OSError, TypeError, ValueError) as exc:
        raise NotImplementedError(
            f"Cannot load filter definition for {name_or_function}"
        ) from exc


def clear_cache() -> None:
    FILTER_CACHE.clear()
