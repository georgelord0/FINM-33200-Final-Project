"""Target variable construction for the src pipeline.

Builds next-day excess returns and binary direction labels used as
supervised-learning targets.  All functions are pure (no side effects)
and shift targets forward to prevent look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import COL_DATE, COL_PERMNO, COL_RET, COL_RF
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------


def build_excess_return_target(
    returns: pd.DataFrame,
    rf: pd.Series,
) -> pd.DataFrame:
    """Compute next-day excess return as the prediction target.

    Parameters
    ----------
    returns:
        DataFrame with columns ``[date, permno, ret]``.  Must be sorted by
        ``[permno, date]``.
    rf:
        Series indexed by date containing the daily risk-free rate.

    Returns
    -------
    pd.DataFrame
        Copy of *returns* with two new columns:

        * ``excess_ret`` -- today's excess return (ret - rf).
        * ``target_ret`` -- **next-day** excess return (shifted forward by 1
          within each permno group).  This is the value we want to predict.
    """
    df = returns.copy()
    df = df.sort_values([COL_PERMNO, COL_DATE]).reset_index(drop=True)

    # Align risk-free rate by date
    rf_aligned = df[COL_DATE].map(rf)
    df["excess_ret"] = df[COL_RET] - rf_aligned

    # Target is the *next* day's excess return -- shift(-1) within each stock
    df["target_ret"] = df.groupby(COL_PERMNO)["excess_ret"].shift(-1)

    log.info(
        "Built excess-return targets: {} rows, {:.1%} non-null targets",
        len(df),
        df["target_ret"].notna().mean(),
    )
    return df


def build_direction_label(target_ret: pd.Series) -> pd.Series:
    """Convert a continuous return target to a binary direction label.

    Parameters
    ----------
    target_ret:
        Series of (excess) returns, typically the ``target_ret`` column
        produced by :func:`build_excess_return_target`.

    Returns
    -------
    pd.Series
        Integer Series: 1 where *target_ret* > 0, 0 otherwise.  ``NaN``
        values in the input propagate as ``NaN`` (float dtype).
    """
    label = (target_ret > 0).astype(float)
    label[target_ret.isna()] = np.nan
    label.name = "direction"
    return label


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load raw CRSP returns and risk-free rate, build targets, and save.

    Reads
    -----
    * ``data/crsp/crsp_daily.parquet``
    * ``data/riskfree/riskfree.parquet``

    Writes
    ------
    * ``data/features/targets.parquet``

    Returns
    -------
    pd.DataFrame
        The targets DataFrame (also saved to disk).
    """
    crsp_path = DATA_DIR / "crsp"
    rf_path = DATA_DIR / "riskfree" / "ff_factors_daily.parquet"

    log.info("Loading CRSP returns from {}", crsp_path)
    crsp = pd.read_parquet(crsp_path)

    log.info("Loading risk-free rate from {}", rf_path)
    rf_df = pd.read_parquet(rf_path)
    rf = rf_df.set_index(COL_DATE)[COL_RF]

    targets = build_excess_return_target(crsp, rf)
    targets["direction"] = build_direction_label(targets["target_ret"])

    out_path = DATA_DIR / "features" / "targets.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(out_path, index=False)
    log.info("Saved targets to {}", out_path)

    return targets


if __name__ == "__main__":
    build()
