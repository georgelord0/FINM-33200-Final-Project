"""Dataset construction modules for the src pipeline."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "build_panel": ("src.datasets.build_panel_dataset", "build_panel"),
    "build_multivariate": ("src.datasets.build_multivariate_dataset", "build_multivariate"),
    "split_by_date": ("src.datasets.build_multivariate_dataset", "split_by_date"),
    "save_dataset": ("src.datasets.build_multivariate_dataset", "save_dataset"),
    "load_dataset": ("src.datasets.build_multivariate_dataset", "load_dataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr = _EXPORTS[name]
    return getattr(import_module(module_name), attr)
