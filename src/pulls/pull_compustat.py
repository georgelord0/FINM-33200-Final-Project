"""Pull and normalize Compustat annual fundamentals from WRDS.

Downloads comp.funda with standard industrial formatting filters,
applies normalization (type casting, book-equity derivation, dedup),
and stores year-partitioned parquet files under data/compustat/.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from src.utils.constants import START_DATE
from src.utils.wrds_connection import get_connection as get_wrds_connection
from src.utils.io import save_parquet as save_partitioned_parquet
from src.utils.logging_utils import get_logger, setup_logger
from src.utils.dates import get_year_ranges as fiscal_years_in_range

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "compustat"

_FUNDA_FIELDS: list[str] = [
    "gvkey",
    "datadate",
    "fyear",
    "at",
    "ceq",
    "lt",
    "sale",
    "ni",
    "oancf",
    "csho",
    "prcc_f",
    "revt",
    "xsga",
    "capx",
    "dp",
    "dltt",
    "dlc",
    "che",
    "act",
    "lct",
    "ppent",
    "invt",
    "rect",
]

_ACCOUNTING_COLS: list[str] = [
    "at",
    "ceq",
    "lt",
    "sale",
    "ni",
    "oancf",
    "csho",
    "prcc_f",
    "revt",
    "xsga",
    "capx",
    "dp",
    "dltt",
    "dlc",
    "che",
    "act",
    "lct",
    "ppent",
    "invt",
    "rect",
]


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class CompustatRowSchema(BaseModel):
    """Validates a single row of the normalised Compustat dataset."""

    gvkey: str
    datadate: dt.date
    fyear: Optional[float] = None
    at: float
    ceq: Optional[float] = None
    lt: Optional[float] = None
    sale: Optional[float] = None
    ni: Optional[float] = None
    oancf: Optional[float] = None
    csho: Optional[float] = None
    prcc_f: Optional[float] = None
    revt: Optional[float] = None
    xsga: Optional[float] = None
    capx: Optional[float] = None
    dp: Optional[float] = None
    dltt: Optional[float] = None
    dlc: Optional[float] = None
    che: Optional[float] = None
    act: Optional[float] = None
    lct: Optional[float] = None
    ppent: Optional[float] = None
    invt: Optional[float] = None
    rect: Optional[float] = None
    book_equity: Optional[float] = None

    model_config = {"strict": False}

    @field_validator("at")
    @classmethod
    def at_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("total assets must be positive")
        return v


def _validate_schema(df: pd.DataFrame, sample_n: int = 500) -> None:
    """Validate a random sample of rows against the pydantic model."""
    n = min(sample_n, len(df))
    sample = df.sample(n=n, random_state=42)
    errors: list[str] = []
    for idx, row in sample.iterrows():
        try:
            CompustatRowSchema(**row.to_dict())
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


def _build_query(year: int) -> str:
    """Build SQL for a single fiscal year from comp.funda."""
    cols = ", ".join(_FUNDA_FIELDS)
    return f"""
        SELECT {cols}
        FROM comp.funda
        WHERE indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          AND datadate BETWEEN '{year}-01-01' AND '{year}-12-31'
        ORDER BY gvkey, datadate
    """


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Compustat data.

    Steps
    -----
    1. Convert ``datadate`` to datetime.
    2. Cast accounting fields to float64.
    3. Remove duplicates on (gvkey, datadate).
    4. Remove rows where ``at`` is null or <= 0.
    5. Compute ``book_equity = ceq`` (fall back to ``at - lt``).
    6. Sort by gvkey, datadate.
    """
    logger.debug(f"Normalizing {len(df):,} Compustat rows")

    # 1. Date conversion
    df["datadate"] = pd.to_datetime(df["datadate"])

    # 2. Cast accounting columns
    for col in _ACCOUNTING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # 3. De-duplicate
    before = len(df)
    df = df.drop_duplicates(subset=["gvkey", "datadate"], keep="last")
    dupes = before - len(df)
    if dupes > 0:
        logger.info(f"  Removed {dupes:,} duplicate (gvkey, datadate) rows")

    # 4. Filter impossible total assets
    df = df[df["at"].notna() & (df["at"] > 0)].copy()
    logger.debug(f"  After at > 0 filter: {len(df):,}")

    # 5. Book equity
    df["book_equity"] = df["ceq"].copy()
    mask_missing_ceq = df["book_equity"].isna()
    df.loc[mask_missing_ceq, "book_equity"] = (
        df.loc[mask_missing_ceq, "at"] - df.loc[mask_missing_ceq, "lt"]
    )
    n_fallback = mask_missing_ceq.sum()
    if n_fallback > 0:
        logger.info(
            f"  Derived book_equity from at-lt for {n_fallback:,} rows"
        )

    # 6. Sort
    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

    # Ensure gvkey is string
    df["gvkey"] = df["gvkey"].astype(str)

    logger.debug(f"  Final normalised row count: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Main pull function
# ---------------------------------------------------------------------------


def pull(
    start_year: int = int(START_DATE[:4]),
    end_year: int | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Pull Compustat annual fundamentals, normalize, and save.

    Parameters
    ----------
    start_year : int
        First calendar year (default from ``constants.START_DATE``).
    end_year : int, optional
        Last calendar year (inclusive). Defaults to current year.
    output_dir : Path, optional
        Override the default output directory.

    Returns
    -------
    pd.DataFrame
        The concatenated, normalised Compustat dataset.
    """
    setup_logger()
    if end_year is None:
        end_year = dt.date.today().year
    if output_dir is None:
        output_dir = _OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pulling Compustat data for {start_year}–{end_year}")

    conn = get_wrds_connection()
    years = list(range(start_year, end_year + 1))
    chunks: list[pd.DataFrame] = []

    for year in tqdm(years, desc="Compustat yearly pulls"):
        cache_file = output_dir / f"year={year}" / "part-0.parquet"
        if cache_file.exists():
            logger.info(f"  {year}: loading from cache")
            chunk = pd.read_parquet(cache_file)
        else:
            query = _build_query(year)
            logger.info(f"  {year}: querying WRDS …")
            raw = conn.query(query, date_cols=["datadate"])

            if raw.empty:
                logger.warning(f"  {year}: no data returned, skipping")
                continue

            chunk = _normalize(raw)

            # Validate schema on first year and every 5 years
            if year == start_year or year % 5 == 0:
                _validate_schema(chunk)

            # Save partitioned
            partition_dir = output_dir / f"year={year}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            chunk.to_parquet(cache_file, index=False, engine="pyarrow")
            logger.info(
                f"  {year}: saved {len(chunk):,} rows → {cache_file}"
            )

        chunks.append(chunk)

    conn.close()

    if not chunks:
        logger.error("No data pulled from Compustat")
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    logger.success(f"Compustat pull complete: {len(result):,} total rows")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pull()
