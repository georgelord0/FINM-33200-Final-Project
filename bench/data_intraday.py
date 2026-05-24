"""
Intraday data dispatcher.

Reads a cfg["data"] block and forwards to the right source loader, returning
the shared Panel type. Source-specific kwargs live under the cfg block; see
configs/intraday_*.yaml for the wire format.

Loaders are imported lazily so a missing optional dep doesn't break runs
that use a different source.
"""

from .protocols import Panel


def load_panel(cfg: dict) -> Panel:
    src = cfg["source"]
    if src == "yfinance_intraday":
        from . import data_yf_intraday
        return data_yf_intraday.load_panel(
            start=cfg["start"], end=cfg["end"],
            cache_dir=cfg["cache_dir"],
            tickers=cfg["tickers"],
            interval=cfg.get("interval", "1m"),
            covariate_kwargs=cfg.get("covariates"),
        )
    if src == "wrds":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from . import data_wrds
        return data_wrds.load_panel(
            start=cfg["start"], end=cfg["end"],
            cache_dir=cfg["cache_dir"],
            tickers=cfg["tickers"],
            interval=cfg.get("interval", "1s"),
            covariate_kwargs=cfg.get("covariates"),
        )
    if src == "synthetic_intraday":
        from . import data_synthetic
        return data_synthetic.load_panel(
            start=cfg["start"], end=cfg["end"],
            n_assets=cfg["n_assets"],
            freq=cfg.get("interval", "1min"),
            seed=cfg.get("seed", 0),
            covariate_kwargs=cfg.get("covariates"),
        )
    raise ValueError(f"unknown intraday source: {src}")
