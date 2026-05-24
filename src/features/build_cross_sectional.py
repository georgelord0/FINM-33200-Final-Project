"""Cross-sectional feature construction for the src pipeline.

Builds market-relative returns, sector-relative returns, cross-sectional
percentile ranks, and cross-sectional z-scores.  These features capture
an asset's position within the cross-section at each point in time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import COL_DATE, COL_MKT_RF, COL_PERMNO, COL_RET
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------


def build_market_relative_return(
    returns: pd.DataFrame,
    market_ret: pd.Series,
) -> pd.DataFrame:
    """Compute each asset's return relative to the market.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]``.
    market_ret:
        Series indexed by date containing the market return.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, market_relative_ret]`` where
        ``market_relative_ret = ret - market_ret``.
    """
    df = returns[[COL_DATE, COL_PERMNO, COL_RET]].copy()
    mkt_aligned = df[COL_DATE].map(market_ret)
    df["market_relative_ret"] = df[COL_RET] - mkt_aligned

    log.info("Built market-relative returns: {} rows", len(df))
    return df[[COL_DATE, COL_PERMNO, "market_relative_ret"]]


def build_sector_relative_return(
    returns: pd.DataFrame,
    sectors: pd.Series,
) -> pd.DataFrame:
    """Compute each asset's return relative to its sector mean.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]``.
    sectors:
        Series indexed by ``permno`` mapping each stock to its sector
        (e.g., SIC code or GICS sector).

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, sector_relative_ret]`` where
        ``sector_relative_ret = ret - sector_mean_ret``.
    """
    df = returns[[COL_DATE, COL_PERMNO, COL_RET]].copy()
    df["_sector"] = df[COL_PERMNO].map(sectors)

    # Compute sector mean return each day
    sector_mean = df.groupby([COL_DATE, "_sector"])[COL_RET].transform("mean")
    df["sector_relative_ret"] = df[COL_RET] - sector_mean

    log.info("Built sector-relative returns: {} rows", len(df))
    return df[[COL_DATE, COL_PERMNO, "sector_relative_ret"]]


def build_rolling_rank(
    feature: pd.DataFrame,
    value_col: str,
    window: int = 252,
) -> pd.DataFrame:
    """Compute rolling cross-sectional percentile rank.

    At each date, ranks all assets by *value_col* and converts to a
    percentile (0 to 1).

    Parameters
    ----------
    feature:
        DataFrame with columns ``[date, permno, <value_col>]``.
    value_col:
        Name of the column to rank.
    window:
        Not used for a single-snapshot rank; included for API consistency
        with time-series rolling operations.  The rank is purely
        cross-sectional at each date.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, {value_col}_rank]`` with percentile
        ranks in [0, 1].
    """
    df = feature[[COL_DATE, COL_PERMNO, value_col]].copy()

    rank_col = f"{value_col}_rank"
    df[rank_col] = df.groupby(COL_DATE)[value_col].rank(pct=True)

    log.info("Built cross-sectional rank for '{}': {} rows", value_col, len(df))
    return df[[COL_DATE, COL_PERMNO, rank_col]]


def build_cross_sectional_zscore(
    feature: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Compute the cross-sectional z-score at each date.

    ``z = (x - cross_sectional_mean) / cross_sectional_std``

    Parameters
    ----------
    feature:
        DataFrame with columns ``[date, permno, <value_col>]``.
    value_col:
        Name of the column to normalise.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, {value_col}_zscore]`` with z-scores
        computed across all assets on each date.
    """
    df = feature[[COL_DATE, COL_PERMNO, value_col]].copy()

    cs_mean = df.groupby(COL_DATE)[value_col].transform("mean")
    cs_std = df.groupby(COL_DATE)[value_col].transform("std")

    zscore_col = f"{value_col}_zscore"
    df[zscore_col] = (df[value_col] - cs_mean) / cs_std.replace(0, np.nan)

    log.info(
        "Built cross-sectional z-score for '{}': {} rows", value_col, len(df)
    )
    return df[[COL_DATE, COL_PERMNO, zscore_col]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load CRSP data, build cross-sectional features, and save.

    Reads
    -----
    * ``data/crsp/crsp_daily.parquet``
    * ``data/factors/ff_daily.parquet``

    Writes
    ------
    * ``data/features/cross_sectional_features.parquet``

    Returns
    -------
    pd.DataFrame
        The cross-sectional features (also saved to disk).
    """
    crsp_path = DATA_DIR / "crsp"
    factors_path = DATA_DIR / "factors" / "factors_daily.parquet"

    log.info("Loading CRSP returns from {}", crsp_path)
    crsp = pd.read_parquet(crsp_path)

    log.info("Loading Fama-French factors from {}", factors_path)
    ff = pd.read_parquet(factors_path)
    mkt_ret = ff.set_index(COL_DATE)[COL_MKT_RF]

    mkt_rel = build_market_relative_return(crsp, mkt_ret)

    # Cross-sectional rank and z-score of returns
    rank_df = build_rolling_rank(crsp, value_col=COL_RET)
    zscore_df = build_cross_sectional_zscore(crsp, value_col=COL_RET)

    features = mkt_rel.merge(
        rank_df, on=[COL_DATE, COL_PERMNO], how="outer"
    ).merge(
        zscore_df, on=[COL_DATE, COL_PERMNO], how="outer"
    )

    out_path = DATA_DIR / "features" / "cross_sectional_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved cross-sectional features to {}", out_path)

    return features


if __name__ == "__main__":
    build()
