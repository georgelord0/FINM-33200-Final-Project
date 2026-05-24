"""Momentum feature construction for the src pipeline.

Builds standard momentum factors (1-month through 12-month), short-term
reversal, and the classic 12-1 momentum signal.  All functions are pure
and shift values to avoid look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.constants import (
    COL_DATE,
    COL_PERMNO,
    COL_RET,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_YEAR,
)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------

# Standard momentum windows (trading days)
_MOM_WINDOWS: dict[str, int] = {
    "mom_1m": 1 * TRADING_DAYS_PER_MONTH,    # 21
    "mom_3m": 3 * TRADING_DAYS_PER_MONTH,    # 63
    "mom_6m": 6 * TRADING_DAYS_PER_MONTH,    # 126
    "mom_12m": TRADING_DAYS_PER_YEAR,         # 252
}


def build_momentum(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute a comprehensive set of momentum features per stock.

    Features produced:

    * ``mom_1m``  -- 21-day cumulative return, shifted by 1.
    * ``mom_3m``  -- 63-day cumulative return, shifted by 1.
    * ``mom_6m``  -- 126-day cumulative return, shifted by 1.
    * ``mom_12m`` -- 252-day cumulative return, shifted by 1.
    * ``short_term_reversal`` -- 5-day cumulative return, shifted by 1.
    * ``mom_12_1`` -- 12-1 momentum: rolling(252).sum().shift(21)
      minus rolling(21).sum().shift(1).  Captures the medium-term
      trend while excluding the most recent month (reversal zone).

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]``, sorted by
        ``[permno, date]``.

    Returns
    -------
    pd.DataFrame
        Identifiers ``[date, permno]`` plus the momentum feature columns.
        All features use only past data (no look-ahead).
    """
    df = returns[[COL_DATE, COL_PERMNO, COL_RET]].copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    grouped = df.groupby(COL_PERMNO)[COL_RET]

    # Standard momentum windows
    for col_name, window in _MOM_WINDOWS.items():
        rolling_sum = grouped.transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
        )
        df[col_name] = df.groupby(COL_PERMNO)[COL_RET].transform(
            lambda x, w=window: x.rolling(window=w, min_periods=w).sum()
        )
        # Shift by 1 to avoid using today's return
        df[col_name] = df.groupby(COL_PERMNO)[col_name].shift(1)

    # Short-term reversal: 5-day return, lagged by 1
    df["short_term_reversal"] = grouped.transform(
        lambda x: x.rolling(window=5, min_periods=5).sum()
    )
    df["short_term_reversal"] = df.groupby(COL_PERMNO)["short_term_reversal"].shift(1)

    # 12-1 momentum: past 12 months excluding the most recent month
    # = rolling(252).sum().shift(21) - rolling(21).sum().shift(1)
    rolling_12m = grouped.transform(
        lambda x: x.rolling(window=252, min_periods=252).sum()
    )
    rolling_1m = grouped.transform(
        lambda x: x.rolling(window=21, min_periods=21).sum()
    )
    df["mom_12_1"] = (
        df.groupby(COL_PERMNO).apply(
            lambda g: rolling_12m.loc[g.index].shift(21), include_groups=False
        ).droplevel(0).sort_index()
        - df.groupby(COL_PERMNO).apply(
            lambda g: rolling_1m.loc[g.index].shift(1), include_groups=False
        ).droplevel(0).sort_index()
    )

    feature_cols = list(_MOM_WINDOWS.keys()) + ["short_term_reversal", "mom_12_1"]
    log.info(
        "Built momentum features: {} columns, {} rows",
        len(feature_cols),
        len(df),
    )
    return df[[COL_DATE, COL_PERMNO] + feature_cols]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load CRSP returns, build momentum features, and save.

    Reads
    -----
    * ``data/crsp/crsp_daily.parquet``

    Writes
    ------
    * ``data/features/momentum_features.parquet``

    Returns
    -------
    pd.DataFrame
        The momentum features (also saved to disk).
    """
    crsp_path = DATA_DIR / "crsp"

    log.info("Loading CRSP returns from {}", crsp_path)
    crsp = pd.read_parquet(crsp_path)

    features = build_momentum(crsp)

    out_path = DATA_DIR / "features" / "momentum_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved momentum features to {}", out_path)

    return features


if __name__ == "__main__":
    build()
