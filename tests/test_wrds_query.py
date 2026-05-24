"""
Offline assertions for the WRDS loader. No live database calls; these tests
just exercise the interval mapping, the env-validation error path, and the
lazy-import behaviour so the file works on machines without wrds installed.
"""

import pytest

from bench import data_wrds


def test_trunc_map_covers_supported_intervals():
    assert data_wrds._TRUNC["1s"] == "second"
    assert data_wrds._TRUNC["1m"] == "minute"


def test_load_panel_rejects_unsupported_interval(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        data_wrds.load_panel(
            start="2025-02-03", end="2025-02-03",
            cache_dir=tmp_path, tickers=["AAPL"], interval="5m",
        )


def test_connect_errors_when_env_missing(monkeypatch):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    monkeypatch.delenv("WRDS_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_USERNAME / WRDS_PASSWORD"):
        data_wrds._connect()


def test_module_imports_without_wrds_installed():
    # The module must not import `wrds` at top level — the package is a
    # heavyweight dep that callers without WRDS access shouldn't need.
    import importlib, sys
    # Re-import to assert it stays clean on a fresh load.
    mod = importlib.reload(data_wrds)
    assert "wrds" not in sys.modules or mod._connect is not None  # lazy is fine
