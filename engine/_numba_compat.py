"""
Thin shim: exposes numba.njit when numba is installed, otherwise a no-op decorator.
All code here is excluded from coverage (it's a pure import-time shim).
"""
try:
    from numba import njit  # pragma: no cover
except ImportError:  # pragma: no cover
    def njit(func=None, **kwargs):  # pragma: no cover
        if func is not None:
            return func
        return lambda f: f

__all__ = ["njit"]
