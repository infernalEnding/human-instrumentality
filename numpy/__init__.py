"""A tiny subset of NumPy used for the kata style exercises.

This repository does not depend on the real `numpy` package at runtime, but
the reference implementation and the accompanying tests were written assuming
NumPy-like semantics for a handful of helpers (array creation, simple maths
and byte conversion).  To keep the project self-contained we ship a very small
pure-Python shim that mimics just enough of the NumPy API for the tests to run.

The goal of the shim is not to be fast or feature complete – it only supports
the dtypes and operations that are exercised in the unit tests.  The module is
documented and intentionally straightforward so it can act as a teaching aid
rather than a drop-in replacement for real NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence
import math
import struct


@dataclass(frozen=True)
class _DType:
    name: str
    pytype: type
    itemsize: int

    def __call__(self, value):  # pragma: no cover - trivial wrapper
        return self.pytype(value)


float32 = _DType("float32", float, 4)
int16 = _DType("int16", int, 2)


class ndarray(Sequence[float]):
    """Very small 1-D oriented array container used in the tests."""

    def __init__(self, data: Iterable[float], *, dtype: _DType = float32, shape: tuple[int, ...] | None = None) -> None:
        values = list(data)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "dtype", dtype)
        if shape is None:
            shape = (len(values),)
        object.__setattr__(self, "shape", shape)

    # Sequence protocol -------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - delegated to list
        return len(self._values)

    def __iter__(self) -> Iterator[float]:  # pragma: no cover - delegated
        return iter(self._values)

    def __getitem__(self, index):  # pragma: no cover - delegated
        return self._values[index]

    # Numeric helpers ----------------------------------------------------
    def astype(self, dtype: _DType) -> "ndarray":
        return ndarray((dtype.pytype(v) for v in self._values), dtype=dtype, shape=self.shape)

    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def reshape(self, *shape: int) -> "ndarray":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if len(shape) == 1 and shape[0] == -1:
            return ndarray(self._values, dtype=self.dtype, shape=(len(self._values),))
        if -1 in shape:
            known = 1
            unknown_index = None
            for idx, value in enumerate(shape):
                if value == -1:
                    unknown_index = idx
                else:
                    known *= value
            if unknown_index is None or known == 0:
                raise ValueError("invalid reshape request")
            inferred = len(self._values) // known
            new_shape = list(shape)
            new_shape[unknown_index] = inferred
            shape = tuple(new_shape)
        total = 1
        for dim in shape:
            total *= dim
        if total != len(self._values):
            raise ValueError("cannot reshape array of size %d into shape %r" % (len(self._values), shape))
        return ndarray(self._values, dtype=self.dtype, shape=tuple(shape))

    def tobytes(self) -> bytes:
        if self.dtype is int16:
            fmt = f"<{len(self._values)}h"
            return struct.pack(fmt, *[int(v) for v in self._values])
        raise TypeError("tobytes only supports int16 arrays in this shim")

    # In-place/standard arithmetic --------------------------------------
    def _binary_op(self, other, op):
        if isinstance(other, ndarray):
            rhs = other._values
        else:
            rhs = [other] * len(self._values)
        return ndarray((op(a, b) for a, b in zip(self._values, rhs)), dtype=self.dtype, shape=self.shape)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __itruediv__(self, other):
        divided = self.__truediv__(other)
        object.__setattr__(self, "_values", divided._values)
        return self

    def __add__(self, other):  # pragma: no cover - unused today
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other):  # pragma: no cover - unused today
        return self._binary_op(other, lambda a, b: a - b)


def array(values: Iterable[float], dtype: _DType | None = None) -> ndarray:
    dtype = dtype or float32
    return ndarray((dtype.pytype(v) for v in values), dtype=dtype)


def linspace(start: float, stop: float, *, num: int, dtype: _DType = float32) -> ndarray:
    if num <= 1:
        return ndarray([dtype.pytype(start)], dtype=dtype)
    step = (stop - start) / (num - 1)
    values = [start + i * step for i in range(num)]
    return ndarray((dtype.pytype(v) for v in values), dtype=dtype)


def clip(data: ndarray, minimum: float, maximum: float) -> ndarray:
    return ndarray((min(max(v, minimum), maximum) for v in data), dtype=data.dtype, shape=data.shape)


def arange(start: int, stop: int | None = None, *, dtype: _DType = int16) -> ndarray:
    if stop is None:
        start, stop = 0, start
    values = list(range(start, stop))
    return ndarray((dtype.pytype(v) for v in values), dtype=dtype)


def zeros(length: int, *, dtype: _DType = float32) -> ndarray:
    return ndarray((0 for _ in range(length)), dtype=dtype)


def ones(length: int, *, dtype: _DType = int16) -> ndarray:
    return ndarray((1 for _ in range(length)), dtype=dtype)


def _chunks(values: list[float], size: int) -> list[list[float]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def mean(values: Iterable[float] | ndarray, axis: int | None = None):
    if isinstance(values, ndarray):
        if axis is None:
            return values.mean()
        if len(values.shape) != 2:
            raise NotImplementedError("axis-aware mean only supports 2-D arrays in this shim")
        rows = _chunks(values._values, values.shape[1])
        if axis == 1:
            data = [sum(row) / len(row) if row else 0.0 for row in rows]
            return ndarray(data, dtype=values.dtype, shape=(len(data),))
        if axis == 0:
            cols = list(zip(*rows))
            data = [sum(col) / len(col) if col else 0.0 for col in cols]
            return ndarray(data, dtype=values.dtype, shape=(len(data),))
        raise NotImplementedError("axis %r is not implemented in this shim" % (axis,))

    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    return total / count if count else 0.0


def isclose(a: float, b: float, *, atol: float = 0.0) -> bool:
    return math.isclose(a, b, abs_tol=atol)


def frombuffer(buffer: bytes, *, dtype: _DType = int16) -> ndarray:
    if dtype is not int16:
        raise TypeError("only int16 buffers are supported in this shim")
    if len(buffer) % dtype.itemsize != 0:
        raise ValueError("buffer size must be a multiple of 2 bytes for int16")
    count = len(buffer) // dtype.itemsize
    fmt = f"<{count}h"
    values = struct.unpack(fmt, buffer)
    return ndarray(values, dtype=int16)


def mean_axis1(values: ndarray) -> ndarray:  # pragma: no cover - compatibility helper
    total = sum(values._values)
    return ndarray([total / len(values._values)], dtype=values.dtype)


__all__ = [
    "array",
    "arange",
    "clip",
    "float32",
    "frombuffer",
    "int16",
    "isclose",
    "linspace",
    "mean",
    "ndarray",
    "ones",
    "zeros",
]
