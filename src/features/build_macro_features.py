"""Macroeconomic feature construction for the src pipeline.

Builds yield spread, inflation change, unemployment change, and
macro-regime indicators from FRED-style macro data.  All features
are shifted by 1 day to prevent look-ahead bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import COL_DATE
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------

_MACRO_FEATURE_COLUMNS = [
    "term_spread",
    "inflation_change",
    "unemployment_change",
    "high_vix",
    "inverted_yield_curve",
    "rising_unemployment",
    "high_inflation",
]


def _empty_macro_features() -> pd.DataFrame:
    cols: dict[str, pd.Series] = {
        COL_DATE: pd.Series(dtype="datetime64[ns]"),
    }
    cols.update({col: pd.Series(dtype="float64") for col in _MACRO_FEATURE_COLUMNS})
    return pd.DataFrame(cols)


def build_yield_spread(macro: pd.DataFrame) -> pd.DataFrame:
    """Compute the term spread (10Y Treasury minus 3M T-bill).

    Parameters
    ----------
    macro:
        DataFrame with columns ``[date, gs10, tb3ms]`` where ``gs10`` is
        the 10-year Treasury yield and ``tb3ms`` is the 3-month T-bill rate.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, term_spread]``, shifted by 1 day.
    """
    df = macro[[COL_DATE]].copy()
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    df["term_spread"] = (macro["gs10"] - macro["tb3ms"]).values
    df["term_spread"] = df["term_spread"].shift(1)

    log.info("Built term spread: {} rows", len(df))
    return df


def build_inflation_change(macro: pd.DataFrame) -> pd.DataFrame:
    """Compute the percentage change in CPI as an inflation proxy.

    Parameters
    ----------
    macro:
        DataFrame with columns ``[date, cpi]``.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, inflation_change]``, shifted by 1 day.
    """
    df = macro[[COL_DATE]].copy()
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    df["inflation_change"] = macro["cpi"].pct_change().values
    df["inflation_change"] = df["inflation_change"].shift(1)

    log.info("Built inflation change: {} rows", len(df))
    return df


def build_unemployment_change(macro: pd.DataFrame) -> pd.DataFrame:
    """Compute the first difference in the unemployment rate.

    Parameters
    ----------
    macro:
        DataFrame with columns ``[date, unrate]``.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, unemployment_change]``, shifted by 1 day.
    """
    df = macro[[COL_DATE]].copy()
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    df["unemployment_change"] = macro["unrate"].diff().values
    df["unemployment_change"] = df["unemployment_change"].shift(1)

    log.info("Built unemployment change: {} rows", len(df))
    return df


def build_macro_regime(macro: pd.DataFrame) -> pd.DataFrame:
    """Construct binary regime indicators from macro data.

    Features:

    * ``high_vix`` -- 1 if VIX is above its expanding median, else 0.
    * ``inverted_yield_curve`` -- 1 if term spread (gs10 - tb3ms) < 0.
    * ``rising_unemployment`` -- 1 if 3-month change in unrate > 0.
    * ``high_inflation`` -- 1 if CPI YoY pct_change > expanding median.

    All indicators are shifted by 1 day.

    Parameters
    ----------
    macro:
        DataFrame with columns ``[date, gs10, tb3ms, vix, unrate, cpi]``.
        Missing columns are silently skipped (the corresponding indicator
        will not appear in the output).

    Returns
    -------
    pd.DataFrame
        Columns ``[date]`` plus the available regime indicators.
    """
    df = macro[[COL_DATE]].copy()
    df = df.sort_values(COL_DATE).reset_index(drop=True)

    # High VIX regime
    if "vix" in macro.columns:
        vix = macro["vix"].values.copy()
        vix_series = pd.Series(vix, index=df.index)
        expanding_median = vix_series.expanding(min_periods=1).median()
        df["high_vix"] = (vix_series > expanding_median).astype(int)
        df["high_vix"] = df["high_vix"].shift(1)

    # Inverted yield curve
    if "gs10" in macro.columns and "tb3ms" in macro.columns:
        spread = macro["gs10"].values - macro["tb3ms"].values
        df["inverted_yield_curve"] = (spread < 0).astype(int)
        df["inverted_yield_curve"] = df["inverted_yield_curve"].shift(1)

    # Rising unemployment (3-month change)
    if "unrate" in macro.columns:
        unrate = pd.Series(macro["unrate"].values, index=df.index)
        df["rising_unemployment"] = (unrate.diff(3) > 0).astype(int)
        df["rising_unemployment"] = df["rising_unemployment"].shift(1)

    # High inflation (above expanding median of 12-month CPI change)
    if "cpi" in macro.columns:
        cpi = pd.Series(macro["cpi"].values, index=df.index)
        cpi_yoy = cpi.pct_change(12)
        expanding_med = cpi_yoy.expanding(min_periods=1).median()
        df["high_inflation"] = (cpi_yoy > expanding_med).astype(int)
        df["high_inflation"] = df["high_inflation"].shift(1)

    regime_cols = [
        c for c in ["high_vix", "inverted_yield_curve",
                     "rising_unemployment", "high_inflation"]
        if c in df.columns
    ]
    log.info("Built macro regime indicators: {}", regime_cols)
    return df[[COL_DATE] + regime_cols]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load macro data, build macro features, and save.

    Reads
    -----
    * ``data/macro/macro.parquet``

    Writes
    ------
    * ``data/features/macro_features.parquet``

    Returns
    -------
    pd.DataFrame
        The macro features (also saved to disk).
    """
    macro_path = DATA_DIR / "macro" / "macro_daily.parquet"

    log.info("Loading macro data from {}", macro_path)
    macro = pd.read_parquet(macro_path)
    macro = macro.sort_values(COL_DATE).reset_index(drop=True)

    if macro.empty:
        log.warning("Macro data is empty; writing empty macro feature file")
        features = _empty_macro_features()
    else:
        spread = build_yield_spread(macro)
        inflation = build_inflation_change(macro)
        unemp = build_unemployment_change(macro)
        regime = build_macro_regime(macro)

        features = spread
        for feat_df in [inflation, unemp, regime]:
            features = features.merge(feat_df, on=COL_DATE, how="outer")

    out_path = DATA_DIR / "features" / "macro_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved macro features to {}", out_path)

    return features


if __name__ == "__main__":
    build()
