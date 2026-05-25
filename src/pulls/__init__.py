"""Data pull modules for WRDS datasets.

Imports are lazy so `python -m src.pulls.<module>` does not pre-import the
target module before runpy executes it.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "pull_crsp": ("src.pulls.pull_crsp", "pull"),
    "pull_compustat": ("src.pulls.pull_compustat", "pull"),
    "pull_linking": ("src.pulls.pull_linking", "pull"),
    "merge_crsp_compustat": ("src.pulls.pull_linking", "merge_crsp_compustat"),
    "pull_riskfree": ("src.pulls.pull_riskfree", "pull"),
    "pull_factors": ("src.pulls.pull_factors", "pull"),
    "pull_macro": ("src.pulls.pull_macro", "pull"),
    "pull_ibes": ("src.pulls.pull_ibes", "pull"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr = _EXPORTS[name]
    return getattr(import_module(module_name), attr)
