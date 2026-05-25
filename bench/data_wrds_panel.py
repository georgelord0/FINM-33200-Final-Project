"""
Loader for the canonical daily WRDS panel built by src.datasets.

The src pipeline writes a long panel keyed by (date, permno), with current-day
excess returns and engineered features. This adapter pivots it into the shared
bench.Panel shape used by the walk-forward harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .protocols import Panel

DATE_COL = "date"
ASSET_COL = "permno"
RETURN_COL = "excess_ret"
MCAP_COL = "market_cap"

_EXCLUDE_EXACT = {
    DATE_COL,
    ASSET_COL,
    "gvkey",
    "ticker",
    "cusip",
    "ncusip",
    "comnam",
    "conm",
    "sector",
    "target_ret",
    "target",
    "direction",
    RETURN_COL,
    "ret",
    "retx",
    "year",
    "shrcd",
    "exchcd",
    "siccd",
    "prc",
    "adj_prc",
    "bid",
    "ask",
    "vol",
    "shrout",
    MCAP_COL,
    "cfacpr",
    "cfacshr",
}

_EXCLUDE_PREFIXES = ("target_", "future_", "lead_")

_CORE_PREFIXES = (
    "ret_lag",
    "cum_ret",
    "log_ret",
    "mom_",
    "short_term_reversal",
    "rvol_",
    "downside_vol_",
    "beta_",
    "turnover",
    "amihud_",
    "volume_zscore_",
    "dollar_volume",
    "book_to_market",
    "roe",
    "roa",
    "gross_profitability",
    "leverage",
    "asset_growth",
    "mktrf",
    "smb",
    "hml",
    "rf",
    "umd",
    "term_spread",
    "inflation_change",
    "unemployment_change",
    "high_vix",
    "inverted_yield_curve",
    "rising_unemployment",
    "high_inflation",
    "market_relative_ret",
    "sector_relative_ret",
    "ret_rank",
    "ret_zscore",
)


def load_panel(
    panel_path: str | Path = "src/data/datasets/panel.parquet",
    start: str | None = None,
    end: str | None = None,
    *,
    top_n_by_mcap: int | None = None,
    min_history_days: int = 252,
    feature_set: str | list[str] = "core",
    oos_year: int | None = None,
) -> Panel:
    """Load a WRDS daily panel as a benchmark Panel.

    Parameters mirror the YAML config keys used by bench.run. `feature_set`
    accepts "core", "all", or an explicit list of feature columns.
    """
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(
            f"WRDS panel not found at {path}. Build it with "
            "`python -m src.datasets.build_panel_dataset` or run "
            "`python scripts/reproduce.py --stage panel`."
        )

    df = pd.read_parquet(path)
    _validate_columns(df, [DATE_COL, ASSET_COL, RETURN_COL])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, ASSET_COL])

    if start is not None:
        df = df[df[DATE_COL] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df[DATE_COL] <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError("WRDS panel is empty after date filtering")

    assets = _select_assets(
        df,
        top_n_by_mcap=top_n_by_mcap,
        min_history_days=min_history_days,
        oos_year=oos_year,
    )
    df = df[df[ASSET_COL].isin(assets)]
    if df.empty:
        raise ValueError("WRDS panel is empty after universe filtering")

    returns = (
        df.pivot(index=DATE_COL, columns=ASSET_COL, values=RETURN_COL)
        .sort_index()
        .sort_index(axis=1)
    )
    feature_cols = _resolve_feature_columns(df, feature_set)
    if not feature_cols:
        raise ValueError(f"feature_set {feature_set!r} selected no covariate columns")
    covariates = (
        df[[DATE_COL, ASSET_COL, *feature_cols]]
        .set_index([DATE_COL, ASSET_COL])
        .sort_index()
    )
    covariates.index.names = ["date", "asset_id"]
    returns.index.name = "date"
    returns.columns.name = "asset_id"

    return Panel(returns=returns, covariates=covariates, freq="B")


def _validate_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"WRDS panel is missing required columns: {missing}")


def _select_assets(
    df: pd.DataFrame,
    *,
    top_n_by_mcap: int | None,
    min_history_days: int,
    oos_year: int | None,
) -> pd.Index:
    counts = df.dropna(subset=[RETURN_COL]).groupby(ASSET_COL)[RETURN_COL].size()
    eligible = counts[counts >= min_history_days].index
    if top_n_by_mcap is None:
        return eligible
    if MCAP_COL not in df.columns:
        raise ValueError("top_n_by_mcap requires a market_cap column")

    cutoff = pd.Timestamp(f"{oos_year - 1}-12-31") if oos_year else df[DATE_COL].max()
    mcap = (
        df[(df[ASSET_COL].isin(eligible)) & (df[DATE_COL] <= cutoff)]
        .dropna(subset=[MCAP_COL])
        .sort_values([ASSET_COL, DATE_COL])
        .groupby(ASSET_COL)[MCAP_COL]
        .last()
        .sort_values(ascending=False)
    )
    assets = mcap.head(top_n_by_mcap).index
    if len(assets) == 0:
        raise ValueError("top_n_by_mcap selected no assets")
    return assets


def _resolve_feature_columns(df: pd.DataFrame, feature_set: str | list[str]) -> list[str]:
    numeric = set(df.select_dtypes(include="number").columns)
    allowed = [
        c for c in df.columns
        if c in numeric and _is_allowed_feature(c)
    ]
    if isinstance(feature_set, list):
        missing = [c for c in feature_set if c not in df.columns]
        if missing:
            raise ValueError(f"requested feature columns not found: {missing}")
        blocked = [c for c in feature_set if c not in allowed]
        if blocked:
            raise ValueError(f"requested feature columns are not allowed: {blocked}")
        return list(feature_set)
    if feature_set == "all":
        return allowed
    if feature_set == "core":
        return [c for c in allowed if c.startswith(_CORE_PREFIXES)]
    raise ValueError("feature_set must be 'core', 'all', or a list of columns")


def _is_allowed_feature(col: str) -> bool:
    if col in _EXCLUDE_EXACT:
        return False
    if col.startswith(_EXCLUDE_PREFIXES):
        return False
    return True
