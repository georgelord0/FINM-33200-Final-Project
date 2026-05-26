"""
yfinance 1-minute loader for the short-horizon smoke run.

Caveats: yfinance gives ~7 days of 1m bars per request and ~30 days of total
history. Don't try to pull a year of 1m data — yfinance will silently truncate.
Use this for the smoke config; use Databento for anything longer / 1-second.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .covariates_intraday import build_intraday_covariates
from .protocols import Panel

VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "1h"}


def _download(tickers: list[str], start: str, end: str, interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        tickers, start=start, end=end, interval=interval, auto_adjust=True,
        progress=False, threads=True, group_by="ticker",
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", level=1, axis=1)
        vol = data.xs("Volume", level=1, axis=1)
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
        vol = data[["Volume"]].rename(columns={"Volume": tickers[0]})
    return close, vol


def load_panel(
    start: str,
    end: str,
    cache_dir: str | Path,
    tickers: list[str],
    interval: str = "1m",
    covariate_kwargs: dict | None = None,
) -> Panel:
    if interval not in VALID_INTERVALS:
        raise ValueError(f"yfinance interval {interval!r} not supported. one of {sorted(VALID_INTERVALS)}")
    cache_dir = Path(cache_dir) / "yf_intraday"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = f"n{len(tickers)}_{interval}"
    price_cache = cache_dir / f"prices_{tag}_{start}_{end}.parquet"
    vol_cache = cache_dir / f"volume_{tag}_{start}_{end}.parquet"

    if price_cache.exists() and vol_cache.exists():
        close = pd.read_parquet(price_cache)
        vol = pd.read_parquet(vol_cache)
    else:
        close, vol = _download(tickers, start, end, interval)
        close.to_parquet(price_cache)
        vol.to_parquet(vol_cache)

    close = close.sort_index().dropna(axis=1, how="all")
    vol = vol.reindex_like(close)

    # ±10% per bar — see data_wrds.py for why ±1 was too loose
    returns = np.log(close).diff().clip(-0.10, 0.10).iloc[1:]
    covariates = build_intraday_covariates(returns, vol.loc[returns.index], **(covariate_kwargs or {}))
    return Panel(returns=returns, covariates=covariates, freq=interval)
