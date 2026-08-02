"""Helpers for importing optional dependencies with consistent error messages."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


def _is_missing_target_module(err: ModuleNotFoundError, module_name: str) -> bool:
    """Return whether the missing module error refers to the requested dependency itself."""
    missing_name = getattr(err, "name", None)
    if missing_name is None:
        return False
    return missing_name == module_name or module_name.startswith(f"{missing_name}.")


def import_optional_dependency(
    module_name: str,
    *,
    attribute: str | None = None,
    package_name: str | None = None,
    feature: str | None = None,
    install_hint: str | None = None,
    error_message: str | None = None,
    raise_on_missing: bool = True,
) -> ModuleType | Any | None:
    """Import an optional dependency and raise a guided error when unavailable."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if not _is_missing_target_module(err, module_name):
            raise

        if not raise_on_missing:
            return None

        if error_message is None:
            dependency_name = package_name or module_name.split(".")[0]
            error_message = f"{dependency_name} is required"
            if feature is not None:
                error_message += f" for {feature}"
            error_message += "."
            if install_hint is not None:
                error_message += f" Please install with `{install_hint}`."

        raise ImportError(error_message) from err
    except ImportError:
        raise

    if attribute is None:
        return module

    try:
        return getattr(module, attribute)
    except AttributeError as err:
        dependency_name = package_name or module_name
        raise ImportError(f"{dependency_name} does not provide `{attribute}`.") from err
