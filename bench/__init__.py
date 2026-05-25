__all__ = ["Forecaster", "Panel"]

_EXPORTS = {
    "Forecaster": ("bench.protocols", "Forecaster"),
    "Panel": ("bench.protocols", "Panel"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr = _EXPORTS[name]
    from importlib import import_module

    return getattr(import_module(module_name), attr)
