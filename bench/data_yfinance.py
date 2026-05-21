"""
yfinance-backed data loader for the exploratory run.

Pulls daily adjusted close + volume for a set of tickers and builds a Panel
with returns and a small covariate set: lagged returns (1, 5, 21d), 21-day
realized vol, 12-1 momentum, log volume z-score. Results are cached to a
local parquet so reruns are cheap.

Caveats (worth flagging in the writeup): current S&P 500 membership has
survivorship bias. Max's CRSP pipeline replaces this loader for the final
submission.
"""

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .protocols import Panel

SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_tickers(cache_dir: Path | None = None) -> list[str]:
    if cache_dir is not None:
        cached = cache_dir / "sp500_tickers.txt"
        if cached.exists():
            return cached.read_text().splitlines()

    # Wikipedia rejects the default urllib User-Agent with 403.
    req = urllib.request.Request(SP500_WIKI, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read()
    df = pd.read_html(io.BytesIO(html), header=0)[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "sp500_tickers.txt").write_text("\n".join(tickers))
    return tickers


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True,
                       progress=False, threads=True, group_by="ticker")
    # yfinance returns a column MultiIndex (ticker, field) when len(tickers)>1.
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", level=1, axis=1)
        vol = data.xs("Volume", level=1, axis=1)
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
        vol = data[["Volume"]].rename(columns={"Volume": tickers[0]})
    return close, vol


def _cache_path(cache_dir: Path, start: str, end: str, tag: str) -> Path:
    return cache_dir / f"prices_{tag}_{start}_{end}.parquet"


def load_panel(
    start: str,
    end: str,
    cache_dir: str | Path,
    tickers: list[str] | None = None,
    top_n: int | None = None,
) -> Panel:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None and top_n is None:
        raise ValueError("must supply either tickers or top_n")
    if tickers is None:
        tickers = sp500_tickers(cache_dir)[:top_n]
    tag = f"n{len(tickers)}"

    cache = _cache_path(cache_dir, start, end, tag)
    if cache.exists():
        close = pd.read_parquet(cache)
        vol = pd.read_parquet(cache.with_name(cache.name.replace("prices_", "volume_")))
    else:
        close, vol = _download(tickers, start, end)
        close.to_parquet(cache)
        vol.to_parquet(cache.with_name(cache.name.replace("prices_", "volume_")))

    close = close.sort_index().dropna(axis=1, how="all")
    vol = vol.reindex_like(close)

    returns = np.log(close).diff()
    returns = returns.clip(-1.0, 1.0)
    returns = returns.iloc[1:]

    covariates = _build_covariates(returns, vol.loc[returns.index])

    return Panel(returns=returns, covariates=covariates, freq="B")


def _build_covariates(returns: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Backward-only features. The covariate row at (t, asset_id) depends solely
    on returns[<= t-1] and volume[<= t-1] for that asset. Every series goes
    through .shift(>=1) or a rolling op whose window ends at t-1. This is what
    the walk-forward driver needs: at prediction time it calls
    covariates.xs(asof), and asof is in the OOS year, so leakage would mean a
    train-set covariate using an OOS observation.
    """
    pieces = []
    for lag in (1, 5, 21):
        pieces.append(returns.shift(lag).stack().rename(f"ret_lag{lag}"))
    pieces.append(returns.rolling(21).std().shift(1).stack().rename("vol_21"))
    mom = returns.rolling(252).sum().shift(21) - returns.rolling(21).sum().shift(1)
    pieces.append(mom.stack().rename("mom_12_1"))
    logv = np.log(volume.replace(0, np.nan))
    z = (logv - logv.rolling(63).mean()) / logv.rolling(63).std()
    pieces.append(z.shift(1).stack().rename("log_vol_z"))

    cov = pd.concat(pieces, axis=1)
    cov.index.names = ["date", "asset_id"]
    return cov
