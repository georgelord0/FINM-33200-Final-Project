"""Pull and normalize the CRSP-Compustat linking table from WRDS.

Downloads crsp.ccmxpf_linktable, filters to usable link types,
and provides a helper function to merge CRSP and Compustat panels.
Stored as a single parquet file under data/linking/.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator

from src.utils.wrds_connection import get_connection as get_wrds_connection
from src.utils.logging_utils import get_logger, setup_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "linking"
_OUTPUT_FILE = _OUTPUT_DIR / "ccm_link.parquet"

_VALID_LINKTYPES: tuple[str, ...] = ("LU", "LC", "LS")
_VALID_LINKPRIMS: tuple[str, ...] = ("P", "C")
_SENTINEL_END_DATE = pd.Timestamp("2099-12-31")


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class LinkingRowSchema(BaseModel):
    """Validates a single row of the normalised linking table."""

    gvkey: str
    permno: int
    linkdt: dt.date
    linkenddt: dt.date
    linktype: str
    linkprim: str

    model_config = {"strict": False}

    @field_validator("linktype")
    @classmethod
    def valid_linktype(cls, v: str) -> str:
        if v not in _VALID_LINKTYPES:
            raise ValueError(f"linktype must be one of {_VALID_LINKTYPES}")
        return v

    @field_validator("linkprim")
    @classmethod
    def valid_linkprim(cls, v: str) -> str:
        if v not in _VALID_LINKPRIMS:
            raise ValueError(f"linkprim must be one of {_VALID_LINKPRIMS}")
        return v


def _validate_schema(df: pd.DataFrame, sample_n: int = 500) -> None:
    """Validate a random sample of rows against the pydantic model."""
    n = min(sample_n, len(df))
    sample = df.sample(n=n, random_state=42)
    errors: list[str] = []
    for idx, row in sample.iterrows():
        try:
            LinkingRowSchema(**row.to_dict())
        except Exception as exc:
            errors.append(f"Row {idx}: {exc}")
    if errors:
        logger.warning(f"Schema validation found {len(errors)} issues in sample")
        for e in errors[:5]:
            logger.warning(e)
    else:
        logger.info(f"Schema validation passed on {n}-row sample")


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_LINK_QUERY = """
    SELECT
        gvkey,
        lpermno AS permno,
        linkdt,
        linkenddt,
        linktype,
        linkprim
    FROM crsp.ccmxpf_linktable
    ORDER BY gvkey, linkdt
"""


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the raw CCM linking table.

    Steps
    -----
    1. Convert date columns to datetime.
    2. Filter ``linktype`` in ('LU', 'LC', 'LS').
    3. Filter ``linkprim`` in ('P', 'C').
    4. Fill missing ``linkenddt`` with 2099-12-31.
    5. Remove duplicate rows.
    6. Sort by gvkey, linkdt.
    """
    logger.debug(f"Normalizing {len(df):,} linking rows")

    # 1. Date conversion
    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])

    # 2. Filter link type
    df = df[df["linktype"].isin(_VALID_LINKTYPES)].copy()
    logger.debug(f"  After linktype filter: {len(df):,}")

    # 3. Filter link prim
    df = df[df["linkprim"].isin(_VALID_LINKPRIMS)].copy()
    logger.debug(f"  After linkprim filter: {len(df):,}")

    # 4. Fill missing end dates
    n_missing = df["linkenddt"].isna().sum()
    df["linkenddt"] = df["linkenddt"].fillna(_SENTINEL_END_DATE)
    if n_missing > 0:
        logger.info(f"  Filled {n_missing:,} missing linkenddt with {_SENTINEL_END_DATE.date()}")

    # 5. De-duplicate
    before = len(df)
    df = df.drop_duplicates()
    dupes = before - len(df)
    if dupes > 0:
        logger.info(f"  Removed {dupes:,} duplicate rows")

    # 6. Sort
    df = df.sort_values(["gvkey", "linkdt"]).reset_index(drop=True)

    # Ensure types
    df["gvkey"] = df["gvkey"].astype(str)
    df["permno"] = df["permno"].astype("int64")

    logger.debug(f"  Final normalised row count: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_crsp_compustat(
    crsp_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    link_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge CRSP and Compustat data using the CCM linking table.

    The merge is performed by:
    1. Inner-joining ``crsp_df`` to ``link_df`` on ``permno``, filtering
       to rows where the CRSP date falls within [linkdt, linkenddt].
    2. Inner-joining the result to ``comp_df`` on ``gvkey``, matching
       each CRSP date to the most recent Compustat fiscal year-end
       that is at least 6 months prior (public-information lag).

    Parameters
    ----------
    crsp_df : pd.DataFrame
        Normalised CRSP daily data (must have ``permno``, ``date``).
    comp_df : pd.DataFrame
        Normalised Compustat fundamentals (must have ``gvkey``, ``datadate``).
    link_df : pd.DataFrame
        Normalised linking table (must have ``gvkey``, ``permno``,
        ``linkdt``, ``linkenddt``).

    Returns
    -------
    pd.DataFrame
        Merged panel keyed on (permno, date) with Compustat fields attached.
    """
    logger.info(
        f"Merging CRSP ({len(crsp_df):,} rows) with Compustat "
        f"({len(comp_df):,} rows) via {len(link_df):,} links"
    )

    # Step 1: CRSP ↔ Link
    merged = crsp_df.merge(link_df, on="permno", how="inner")
    merged = merged[
        (merged["date"] >= merged["linkdt"])
        & (merged["date"] <= merged["linkenddt"])
    ].copy()
    logger.info(f"  After link-date filter: {len(merged):,} rows")

    # Step 2: Attach Compustat
    # For each CRSP row, find the most recent Compustat observation
    # at least 6 months (183 days) before the CRSP date.
    comp_df = comp_df.copy()
    comp_df["datadate"] = pd.to_datetime(comp_df["datadate"])

    merged = merged.merge(comp_df, on="gvkey", how="inner", suffixes=("", "_comp"))

    # Public-information lag: datadate must be at least 6 months before the
    # CRSP date but no more than 18 months (stale data guard).
    lag_min = pd.Timedelta(days=183)
    lag_max = pd.Timedelta(days=548)  # ~18 months
    merged["_lag"] = merged["date"] - merged["datadate"]
    merged = merged[
        (merged["_lag"] >= lag_min) & (merged["_lag"] <= lag_max)
    ].copy()

    # Keep the most recent Compustat observation for each (permno, date)
    merged = merged.sort_values("datadate", ascending=False)
    merged = merged.drop_duplicates(subset=["permno", "date"], keep="first")
    merged = merged.drop(columns=["_lag"])

    merged = merged.sort_values(["permno", "date"]).reset_index(drop=True)
    logger.success(f"  Merged dataset: {len(merged):,} rows")

    return merged


# ---------------------------------------------------------------------------
# Main pull function
# ---------------------------------------------------------------------------


def pull(output_dir: Path | None = None) -> pd.DataFrame:
    """Pull the CCM linking table from WRDS, normalize, and save.

    Parameters
    ----------
    output_dir : Path, optional
        Override the default output directory.

    Returns
    -------
    pd.DataFrame
        The normalised linking table.
    """
    setup_logger()
    if output_dir is None:
        output_dir = _OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ccm_link.parquet"

    if output_file.exists():
        logger.info(f"Loading cached linking table from {output_file}")
        return pd.read_parquet(output_file)

    logger.info("Pulling CCM linking table from WRDS")
    conn = get_wrds_connection()
    raw = conn.query(_LINK_QUERY, date_cols=["linkdt", "linkenddt"])
    conn.close()

    if raw.empty:
        logger.error("No data returned for linking table")
        return pd.DataFrame()

    result = _normalize(raw)
    _validate_schema(result)

    result.to_parquet(output_file, index=False, engine="pyarrow")
    logger.success(f"Linking table saved: {len(result):,} rows → {output_file}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pull()
