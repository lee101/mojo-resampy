"""ctypes bridge to the compiled Mojo resampling kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "resample.mojo")
LIB = os.environ.get("MOJO_RESAMPY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-resampy.so"
)

I = ctypes.c_int64
F = ctypes.c_double
_ARGS = [I] * 10 + [F, F, F, I, I, I]


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_RESAMPY_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    if not os.path.exists(script):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no library at {LIB} and no build script at {script}")
    shell = shutil.which("bash")
    if not shell:
        raise BuildError("bash not found")
    proc = subprocess.run(
        [shell, script], cwd=ROOT, capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name in (
            "mrs_resample_f32",
            "mrs_resample_f64",
            "mrs_resample_f64_to_f32",
        ):
            fn = getattr(_library, name)
            fn.argtypes = _ARGS
            fn.restype = None
    return _library


def addr(array: np.ndarray) -> int:
    if array.size == 0:
        raise ValueError("cannot pass an empty array across the Mojo FFI boundary")
    if not array.flags.c_contiguous:
        raise ValueError("Mojo FFI buffers must be C-contiguous")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null pointer across the Mojo FFI boundary")
    return address
