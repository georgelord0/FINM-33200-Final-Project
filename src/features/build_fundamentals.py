"""Fundamental feature construction for the src pipeline.

Builds book-to-market, profitability, leverage, and investment features
from Compustat quarterly data merged with CRSP market-cap data.

IMPORTANT: All fundamentals are lagged by at least 90 calendar days
from ``datadate`` to prevent look-ahead bias, reflecting the typical
SEC filing delay for 10-Q/10-K reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import (
    COL_AT,
    COL_CEQQ,
    COL_DATADATE,
    COL_DATE,
    COL_GVKEY,
    COL_MARKET_CAP,
    COL_NIQ,
    COL_PERMNO,
)
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Number of calendar days to lag fundamentals (SEC filing delay)
FUNDAMENTAL_LAG_DAYS: int = 90

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_availability_date(comp: pd.DataFrame) -> pd.DataFrame:
    """Add an ``avail_date`` column = datadate + 90 calendar days.

    This is the earliest date at which the fundamental data could
    realistically have been known to the market.

    Parameters
    ----------
    comp:
        Compustat quarterly DataFrame with a ``datadate`` column.

    Returns
    -------
    pd.DataFrame
        Copy with an ``avail_date`` column appended.
    """
    df = comp.copy()
    df["avail_date"] = pd.to_datetime(df[COL_DATADATE]) + pd.Timedelta(
        days=FUNDAMENTAL_LAG_DAYS
    )
    return df


# ---------------------------------------------------------------------------
# Pure feature functions
# ---------------------------------------------------------------------------


def build_book_to_market(
    comp: pd.DataFrame,
    crsp: pd.DataFrame,
) -> pd.DataFrame:
    """Compute book-to-market ratio, properly lagged.

    Book equity is taken from Compustat (``ceqq``).  Market capitalisation
    comes from CRSP.  The ratio is lagged by 90 days from ``datadate`` to
    prevent look-ahead bias.

    Parameters
    ----------
    comp:
        Compustat quarterly DataFrame with columns including
        ``[gvkey, permno, datadate, ceqq]``.
    crsp:
        CRSP daily DataFrame with columns ``[date, permno, market_cap]``.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, book_to_market]`` at daily frequency,
        forward-filled from each quarterly observation's availability date.
    """
    cq = _add_availability_date(comp)
    cq = cq[[COL_PERMNO, "avail_date", COL_CEQQ]].dropna(subset=[COL_CEQQ])
    cq = cq.rename(columns={"avail_date": COL_DATE})
    cq[COL_DATE] = pd.to_datetime(cq[COL_DATE])
    cq = cq.sort_values(COL_DATE).reset_index(drop=True)

    # Get market cap from CRSP
    mkt = crsp[[COL_DATE, COL_PERMNO, COL_MARKET_CAP]].copy()
    mkt[COL_DATE] = pd.to_datetime(mkt[COL_DATE])
    mkt = mkt.sort_values(COL_DATE).reset_index(drop=True)

    # Merge-asof: for each fundamental availability date, find the most
    # recent market cap observation
    cq = pd.merge_asof(
        cq,
        mkt,
        on=COL_DATE,
        by=COL_PERMNO,
        direction="backward",
    )

    cq["book_to_market"] = cq[COL_CEQQ] / cq[COL_MARKET_CAP].replace(0, np.nan)

    # Now forward-fill to daily frequency using the CRSP date grid
    daily_grid = mkt[[COL_DATE, COL_PERMNO]].drop_duplicates()
    result = pd.merge_asof(
        daily_grid.sort_values(COL_DATE).reset_index(drop=True),
        cq[[COL_DATE, COL_PERMNO, "book_to_market"]].sort_values(
            COL_DATE
        ).reset_index(drop=True),
        on=COL_DATE,
        by=COL_PERMNO,
        direction="backward",
    )

    log.info(
        "Built book-to-market: {} rows, {:.1%} non-null",
        len(result),
        result["book_to_market"].notna().mean(),
    )
    return result[[COL_DATE, COL_PERMNO, "book_to_market"]]


def build_profitability(comp: pd.DataFrame) -> pd.DataFrame:
    """Compute profitability ratios from Compustat quarterly data.

    Features:

    * ``roe`` -- Return on equity = niq / ceqq
    * ``roa`` -- Return on assets = niq / atq
    * ``gross_profitability`` -- (saleq - cogsq) / atq  (if available)

    All values are lagged by 90 days from ``datadate``.

    Parameters
    ----------
    comp:
        Compustat quarterly DataFrame with columns including
        ``[gvkey, permno, datadate, niq, ceqq, atq]`` and optionally
        ``[saleq, cogsq]``.

    Returns
    -------
    pd.DataFrame
        Columns ``[avail_date, permno, roe, roa, gross_profitability]``.
    """
    cq = _add_availability_date(comp)

    # ROE = net income / common equity
    cq["roe"] = cq[COL_NIQ] / cq[COL_CEQQ].replace(0, np.nan)

    # ROA = net income / total assets
    atq_col = "atq" if "atq" in cq.columns else COL_AT
    cq["roa"] = cq[COL_NIQ] / cq[atq_col].replace(0, np.nan)

    # Gross profitability (if sale and cogs columns exist)
    if "saleq" in cq.columns and "cogsq" in cq.columns:
        cq["gross_profitability"] = (
            (cq["saleq"] - cq["cogsq"]) / cq[atq_col].replace(0, np.nan)
        )
    elif "revtq" in cq.columns and "cogsq" in cq.columns:
        cq["gross_profitability"] = (
            (cq["revtq"] - cq["cogsq"]) / cq[atq_col].replace(0, np.nan)
        )
    else:
        cq["gross_profitability"] = np.nan

    out_cols = [COL_PERMNO, "avail_date", "roe", "roa", "gross_profitability"]
    result = cq[[c for c in out_cols if c in cq.columns]].copy()
    result = result.rename(columns={"avail_date": COL_DATE})

    log.info("Built profitability ratios: {} rows", len(result))
    return result


def build_leverage(comp: pd.DataFrame) -> pd.DataFrame:
    """Compute financial leverage from Compustat quarterly data.

    ``leverage = (dlttq + dlcq) / atq``

    Lagged by 90 days from ``datadate``.

    Parameters
    ----------
    comp:
        Compustat quarterly DataFrame with columns including
        ``[permno, datadate, dlttq, dlcq, atq]``.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, leverage]``.
    """
    cq = _add_availability_date(comp)

    dlttq = cq["dlttq"].fillna(0) if "dlttq" in cq.columns else 0
    dlcq = cq["dlcq"].fillna(0) if "dlcq" in cq.columns else 0
    atq_col = "atq" if "atq" in cq.columns else COL_AT

    cq["leverage"] = (dlttq + dlcq) / cq[atq_col].replace(0, np.nan)

    result = cq[[COL_PERMNO, "avail_date", "leverage"]].copy()
    result = result.rename(columns={"avail_date": COL_DATE})

    log.info("Built leverage: {} rows", len(result))
    return result


def build_investment(comp: pd.DataFrame) -> pd.DataFrame:
    """Compute asset growth (investment) from Compustat quarterly data.

    ``asset_growth = atq / atq_lag4 - 1``  (year-over-year quarterly growth).

    Lagged by 90 days from ``datadate``.

    Parameters
    ----------
    comp:
        Compustat quarterly DataFrame with columns including
        ``[gvkey, permno, datadate, atq]``, sorted by ``[permno, datadate]``.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, permno, asset_growth]``.
    """
    cq = _add_availability_date(comp)
    cq = cq.sort_values([COL_PERMNO, COL_DATADATE]).reset_index(drop=True)

    atq_col = "atq" if "atq" in cq.columns else COL_AT

    # Lag by 4 quarters within each firm
    cq["_atq_lag4"] = cq.groupby(COL_PERMNO)[atq_col].shift(4)
    cq["asset_growth"] = cq[atq_col] / cq["_atq_lag4"].replace(0, np.nan) - 1

    result = cq[[COL_PERMNO, "avail_date", "asset_growth"]].copy()
    result = result.rename(columns={"avail_date": COL_DATE})

    log.info("Built investment (asset growth): {} rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build() -> pd.DataFrame:
    """Load Compustat and CRSP, build fundamental features, and save.

    Reads
    -----
    * ``data/compustat/compustat_quarterly.parquet``
    * ``data/crsp/crsp_daily.parquet``

    Writes
    ------
    * ``data/features/fundamental_features.parquet``

    Returns
    -------
    pd.DataFrame
        The fundamental features (also saved to disk).
    """
    comp_path = DATA_DIR / "compustat"
    crsp_path = DATA_DIR / "crsp"
    link_path = DATA_DIR / "linking" / "ccm_link.parquet"

    log.info("Loading Compustat from {}", comp_path)
    comp = pd.read_parquet(comp_path)

    log.info("Loading CRSP from {}", crsp_path)
    crsp = pd.read_parquet(crsp_path)

    # Map annual column names to quarterly names expected by feature functions
    _annual_to_quarterly = {
        "ceq": "ceqq", "ni": "niq", "at": "atq",
        "sale": "saleq", "revt": "revtq",
        "dltt": "dlttq", "dlc": "dlcq",
    }
    for annual, quarterly in _annual_to_quarterly.items():
        if annual in comp.columns and quarterly not in comp.columns:
            comp[quarterly] = comp[annual]

    # Merge permno from linking table if not present
    if COL_PERMNO not in comp.columns:
        log.info("Loading linking table from {}", link_path)
        link = pd.read_parquet(link_path)
        comp = comp.merge(
            link[[COL_GVKEY, COL_PERMNO, "linkdt", "linkenddt"]],
            on=COL_GVKEY,
            how="inner",
        )
        # Keep rows where datadate falls within link validity
        mask = (comp[COL_DATADATE] >= comp["linkdt"]) & (
            comp[COL_DATADATE] <= comp["linkenddt"]
        )
        comp = comp.loc[mask].drop(columns=["linkdt", "linkenddt"])
        comp = comp.drop_duplicates(subset=[COL_GVKEY, COL_PERMNO, COL_DATADATE])
        log.info("After linking: {} rows with permno", len(comp))

    # Ensure permno dtype consistency between comp and crsp
    comp[COL_PERMNO] = comp[COL_PERMNO].astype("int64")
    crsp[COL_PERMNO] = crsp[COL_PERMNO].astype("int64")

    bm = build_book_to_market(comp, crsp)
    prof = build_profitability(comp)
    lev = build_leverage(comp)
    inv = build_investment(comp)

    # Merge all fundamental features
    features = bm
    for feat_df in [prof, lev, inv]:
        features = features.merge(
            feat_df, on=[COL_DATE, COL_PERMNO], how="outer"
        )

    out_path = DATA_DIR / "features" / "fundamental_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    log.info("Saved fundamental features to {}", out_path)

    return features


if __name__ == "__main__":
    build()
