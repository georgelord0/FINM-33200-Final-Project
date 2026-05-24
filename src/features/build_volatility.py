"""Volatility and risk feature construction for the src pipeline.

Builds realized volatility, downside volatility, and rolling beta estimates.
All functions are pure and shift results by 1 day to prevent look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import (
    COL_DATE,
    COL_MKT_RF,
    COL_PERMNO,
    COL_RET,
    TRADING_DAYS_PER_YEAR,
)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------


def build_realized_vol(
    returns: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling realized volatility for each stock.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]`` sorted by
        ``[permno, date]``.
    windows:
        Rolling-window sizes in trading days.  Defaults to ``[21, 63]``.

    Returns
    -------
    pd.DataFrame
        Columns ``rvol_{w}`` for each window, plus identifiers.
        Values are the rolling standard deviation of returns, shifted
        by 1 day so only past data is used.
    """
    if windows is None:
        windows = [21, 63]

    df = returns[[COL_DATE, COL_PERMNO, COL_RET]].copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    grouped = df.groupby(COL_PERMNO)[COL_RET]

    for w in windows:
        col = f"rvol_{w}"
        df[col] = grouped.transform(
            lambda x, win=w: x.rolling(window=win, min_periods=win).std()
        )
        df[col] = df.groupby(COL_PERMNO)[col].shift(1)

    feature_cols = [f"rvol_{w}" for w in windows]
    log.info("Built realized volatility for windows={}", windows)
    return df[[COL_DATE, COL_PERMNO] + feature_cols]


def build_downside_vol(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Compute rolling downside volatility for each stock.

    Downside volatility is the standard deviation of ``min(ret, 0)`` --
    only negative returns contribute.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]`` sorted by
        ``[permno, date]``.
    window:
        Rolling-window size in trading days.

    Returns
    -------
    pd.DataFrame
        Column ``downside_vol_{window}`` plus identifiers, shifted by 1.
    """
    df = returns[[COL_DATE, COL_PERMNO, COL_RET]].copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    # Only keep the negative part of returns
    df["_neg_ret"] = df[COL_RET].clip(upper=0.0)

    col = f"downside_vol_{window}"
    df[col] = df.groupby(COL_PERMNO)["_neg_ret"].transform(
        lambda x: x.rolling(window=window, min_periods=window).std()
    )
    df[col] = df.groupby(COL_PERMNO)[col].shift(1)

    log.info("Built downside volatility (window={})", window)
    return df[[COL_DATE, COL_PERMNO, col]]


def build_beta(
    returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Compute rolling CAPM beta for each stock.

    Beta = Cov(r_i, r_m) / Var(r_m) estimated over a rolling window.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]`` sorted by
        ``[permno, date]``.
    market_returns:
        Series indexed by date containing the market (excess) return.
    window:
        Rolling-window size in trading days.

    Returns
    -------
    pd.DataFrame
        Column ``beta_{window}`` plus identifiers, shifted by 1.
    """
    df = returns[[COL_DATE, COL_PERMNO, COL_RET]].copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    # Align market return to each row's date
    df["_mkt"] = df[COL_DATE].map(market_returns)

    def _rolling_beta(group: pd.DataFrame) -> pd.Series:
        """Compute rolling beta within a single permno group."""
        ret = group[COL_RET]
        mkt = group["_mkt"]
        cov = ret.rolling(window=window, min_periods=window).cov(mkt)
        var = mkt.rolling(window=window, min_periods=window).var()
        return cov / var

    col = f"beta_{window}"
    df[col] = df.groupby(COL_PERMNO, group_keys=False).apply(_rolling_beta)
    df[col] = df.groupby(COL_PERMNO)[col].shift(1)

    log.info("Built rolling beta (window={})", window)
    return df[[COL_DATE, COL_PERMNO, col]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load CRSP returns and market factor, build volatility features, save.

    Reads
    -----
    * ``data/crsp/crsp_daily.parquet``
    * ``data/factors/ff_daily.parquet``

    Writes
    ------
    * ``data/features/volatility_features.parquet``

    Returns
    -------
    pd.DataFrame
        The volatility features (also saved to disk).
    """
    crsp_path = DATA_DIR / "crsp"
    factors_path = DATA_DIR / "factors" / "factors_daily.parquet"

    log.info("Loading CRSP returns from {}", crsp_path)
    crsp = pd.read_parquet(crsp_path)

    log.info("Loading Fama-French factors from {}", factors_path)
    ff = pd.read_parquet(factors_path)
    mkt_rf = ff.set_index(COL_DATE)[COL_MKT_RF]

    rvol = build_realized_vol(crsp)
    dvol = build_downside_vol(crsp)
    beta = build_beta(crsp, mkt_rf)

    features = rvol.merge(
        dvol, on=[COL_DATE, COL_PERMNO], how="outer"
    ).merge(
        beta, on=[COL_DATE, COL_PERMNO], how="outer"
    )

    out_path = DATA_DIR / "features" / "volatility_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved volatility features to {}", out_path)

    return features


if __name__ == "__main__":
    build()
