"""Liquidity feature construction for the src pipeline.

Builds turnover, Amihud illiquidity, dollar volume, and volume z-score
features.  All functions are pure and shifted to prevent look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import (
    COL_DATE,
    COL_PERMNO,
    COL_PRC,
    COL_RET,
    COL_SHROUT,
    COL_VOL,
)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------


def build_dollar_volume(
    price: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Compute dollar trading volume.

    Parameters
    ----------
    price:
        DataFrame with columns ``[date, permno, prc]``.
    volume:
        DataFrame with columns ``[date, permno, vol]``.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, dollar_volume]`` where
        ``dollar_volume = |prc| * vol``.
    """
    df = price[[COL_DATE, COL_PERMNO, COL_PRC]].copy()
    df[COL_VOL] = volume[COL_VOL].values
    df["dollar_volume"] = df[COL_PRC].abs() * df[COL_VOL]
    log.info("Built dollar volume for {} rows", len(df))
    return df[[COL_DATE, COL_PERMNO, "dollar_volume"]]


def build_turnover(
    volume: pd.DataFrame,
    shrout: pd.DataFrame,
) -> pd.DataFrame:
    """Compute share turnover ratio.

    Parameters
    ----------
    volume:
        DataFrame with columns ``[date, permno, vol]``.
    shrout:
        DataFrame with columns ``[date, permno, shrout]`` (shares outstanding).

    Returns
    -------
    pd.DataFrame
        Column ``turnover`` = vol / shrout, shifted by 1 day.
    """
    df = volume[[COL_DATE, COL_PERMNO, COL_VOL]].copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)
    df[COL_SHROUT] = shrout[COL_SHROUT].values

    df["turnover"] = df[COL_VOL] / df[COL_SHROUT].replace(0, np.nan)
    df["turnover"] = df.groupby(COL_PERMNO)["turnover"].shift(1)

    log.info("Built turnover for {} rows", len(df))
    return df[[COL_DATE, COL_PERMNO, "turnover"]]


def build_amihud(
    returns: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """Compute the Amihud (2002) illiquidity measure.

    ILLIQ = rolling mean of |ret| / dollar_volume.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]``.
    dollar_volume:
        DataFrame with columns ``[date, permno, dollar_volume]``.
    window:
        Rolling-window size in trading days.

    Returns
    -------
    pd.DataFrame
        Column ``amihud_{window}`` shifted by 1 day.
    """
    df = returns[[COL_DATE, COL_PERMNO, COL_RET]].copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)
    df["dollar_volume"] = dollar_volume["dollar_volume"].values

    # Ratio: |return| / dollar volume (avoid division by zero)
    df["_ratio"] = df[COL_RET].abs() / df["dollar_volume"].replace(0, np.nan)

    col = f"amihud_{window}"
    df[col] = df.groupby(COL_PERMNO)["_ratio"].transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )
    df[col] = df.groupby(COL_PERMNO)[col].shift(1)

    log.info("Built Amihud illiquidity (window={})", window)
    return df[[COL_DATE, COL_PERMNO, col]]


def build_volume_zscore(
    volume: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """Compute a z-score of log trading volume.

    The z-score is computed relative to the rolling mean and std of
    log(volume), providing a normalised measure of abnormal volume.

    Parameters
    ----------
    volume:
        DataFrame with columns ``[date, permno, vol]``.
    window:
        Rolling-window size in trading days.

    Returns
    -------
    pd.DataFrame
        Column ``volume_zscore_{window}`` shifted by 1 day.
    """
    df = volume[[COL_DATE, COL_PERMNO, COL_VOL]].copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    df["_log_vol"] = np.log1p(df[COL_VOL].clip(lower=0))

    grouped = df.groupby(COL_PERMNO)["_log_vol"]
    rolling_mean = grouped.transform(
        lambda x: x.rolling(window=window, min_periods=window).mean()
    )
    rolling_std = grouped.transform(
        lambda x: x.rolling(window=window, min_periods=window).std()
    )

    col = f"volume_zscore_{window}"
    df[col] = (df["_log_vol"] - rolling_mean) / rolling_std.replace(0, np.nan)
    df[col] = df.groupby(COL_PERMNO)[col].shift(1)

    log.info("Built volume z-score (window={})", window)
    return df[[COL_DATE, COL_PERMNO, col]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load CRSP data, build liquidity features, and save.

    Reads
    -----
    * ``data/crsp/crsp_daily.parquet``

    Writes
    ------
    * ``data/features/liquidity_features.parquet``

    Returns
    -------
    pd.DataFrame
        The liquidity features (also saved to disk).
    """
    crsp_path = DATA_DIR / "crsp"

    log.info("Loading CRSP data from {}", crsp_path)
    crsp = pd.read_parquet(crsp_path)
    crsp = crsp.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    dvol = build_dollar_volume(crsp, crsp)
    turnover = build_turnover(crsp, crsp)
    amihud = build_amihud(crsp, dvol)
    vol_z = build_volume_zscore(crsp)

    features = turnover.merge(
        amihud, on=[COL_DATE, COL_PERMNO], how="outer"
    ).merge(
        dvol, on=[COL_DATE, COL_PERMNO], how="outer"
    ).merge(
        vol_z, on=[COL_DATE, COL_PERMNO], how="outer"
    )

    out_path = DATA_DIR / "features" / "liquidity_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved liquidity features to {}", out_path)

    return features


if __name__ == "__main__":
    build()
