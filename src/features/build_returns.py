"""Return-based feature construction for the src pipeline.

Builds lagged returns, cumulative return windows, and log returns.
All functions are pure, take DataFrames as input and return DataFrames.
Shifts are applied to prevent look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import COL_DATE, COL_PERMNO, COL_PRC, COL_RET
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------


def build_lagged_returns(
    returns: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create lagged-return features for each stock.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]`` sorted by
        ``[permno, date]``.
    lags:
        List of lag periods (in trading days).  Defaults to ``[1, 5, 21]``.

    Returns
    -------
    pd.DataFrame
        One column ``ret_lag_{k}`` per lag, indexed identically to *returns*.
        Each value is the return observed *k* days **before** the current row,
        so no future information is used.
    """
    if lags is None:
        lags = [1, 5, 21]

    df = returns.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)
    result = pd.DataFrame(index=df.index)

    grouped = df.groupby(COL_PERMNO)[COL_RET]
    for lag in lags:
        result[f"ret_lag_{lag}"] = grouped.shift(lag)

    log.info("Built lagged returns for lags={}", lags)
    return result


def build_cumulative_returns(
    returns: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling cumulative returns over various windows.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]`` sorted by
        ``[permno, date]``.
    windows:
        Rolling-window sizes in trading days.  Defaults to ``[5, 21, 63]``.

    Returns
    -------
    pd.DataFrame
        One column ``cum_ret_{w}`` per window.  Each value is the sum of
        returns over the preceding *w* days, shifted by 1 to avoid
        leaking the current day's return.
    """
    if windows is None:
        windows = [5, 21, 63]

    df = returns.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)
    result = pd.DataFrame(index=df.index)

    grouped = df.groupby(COL_PERMNO)[COL_RET]
    for w in windows:
        col = f"cum_ret_{w}"
        rolling_sum = grouped.transform(
            lambda x, win=w: x.rolling(window=win, min_periods=win).sum()
        )
        # Shift by 1 so we only use information available *before* today
        result[col] = rolling_sum.groupby(df[COL_PERMNO]).shift(1)

    log.info("Built cumulative returns for windows={}", windows)
    return result


def build_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price panel.

    Parameters
    ----------
    prices:
        DataFrame with columns ``[date, permno, prc]`` sorted by
        ``[permno, date]``.  Prices should be absolute values (unsigned).

    Returns
    -------
    pd.DataFrame
        Single column ``log_ret`` = ln(P_t) - ln(P_{t-1}), computed
        within each permno group.
    """
    df = prices.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)
    result = pd.DataFrame(index=df.index)
    result["log_ret"] = np.log(df[COL_PRC].abs()).groupby(df[COL_PERMNO]).diff()

    log.info("Built log returns for {} rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load CRSP data, build return features, and save.

    Reads
    -----
    * ``data/crsp/crsp_daily.parquet``

    Writes
    ------
    * ``data/features/return_features.parquet``

    Returns
    -------
    pd.DataFrame
        The combined return features (also saved to disk).
    """
    crsp_path = DATA_DIR / "crsp"

    log.info("Loading CRSP returns from {}", crsp_path)
    crsp = pd.read_parquet(crsp_path)
    crsp = crsp.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    lagged = build_lagged_returns(crsp)
    cumulative = build_cumulative_returns(crsp)
    log_ret = build_log_returns(crsp)

    features = pd.concat(
        [crsp[[COL_DATE, COL_PERMNO]], lagged, cumulative, log_ret],
        axis=1,
    )

    out_path = DATA_DIR / "features" / "return_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved return features to {}", out_path)

    return features


if __name__ == "__main__":
    build()
