"""Helpers for deprecating public APIs."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=Callable[..., object])

try:
    from deprecation import deprecated as _deprecated
except Exception:  # pragma: no cover - optional dependency
    _deprecated = None


def deprecated(*args, **kwargs):
    """Return a deprecation decorator.

    If the optional `deprecation` dependency is installed, this proxies to
    `deprecation.deprecated`. Otherwise it falls back to a lightweight wrapper
    that emits a DeprecationWarning at call time.
    """
    if _deprecated is not None:
        return _deprecated(*args, **kwargs)

    details = kwargs.get("details", "This function is deprecated and will be removed in a future release.")

    def _decorator(func: T) -> T:
        def _wrapped(*f_args, **f_kwargs):
            warnings.warn(details, DeprecationWarning, stacklevel=2)
            return func(*f_args, **f_kwargs)

        return _wrapped  # type: ignore[return-value]

    return _decorator
