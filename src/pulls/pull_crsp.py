"""Pull and normalize CRSP daily stock data from WRDS.

Downloads daily stock file (crsp.dsf) joined with name history
(crsp.dsenames), applies standard filters and normalization, then
stores the result as year-partitioned parquet files under data/crsp/.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from src.utils.constants import (
    COMMON_STOCK_SHRCDS,
    MAJOR_EXCHANGES,
    START_DATE,
)
from src.utils.wrds_connection import get_connection as get_wrds_connection
from src.utils.io import save_parquet as save_partitioned_parquet
from src.utils.logging_utils import get_logger, setup_logger

logger = get_logger(__name__)
from src.utils.dates import get_year_ranges as fiscal_years_in_range

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "crsp"

_CRSP_COLUMNS = [
    "a.permno",
    "a.date",
    "a.ret",
    "a.retx",
    "a.prc",
    "a.vol",
    "a.shrout",
    "a.cfacpr",
    "a.cfacshr",
    "b.siccd",
    "b.exchcd",
    "b.shrcd",
]

_NUMERIC_COLS = ["ret", "retx", "prc", "vol", "shrout", "cfacpr", "cfacshr"]

_MAX_RETURN = 3.0  # cap: |ret| > 3.0 → NaN


# ---------------------------------------------------------------------------
# Pydantic schema for validation
# ---------------------------------------------------------------------------


class CRSPRowSchema(BaseModel):
    """Validates a single row of the normalised CRSP dataset."""

    permno: int
    date: dt.date
    ret: Optional[float] = None
    retx: Optional[float] = None
    prc: float
    vol: Optional[float] = None
    shrout: float
    cfacpr: Optional[float] = None
    cfacshr: Optional[float] = None
    siccd: Optional[int] = None
    exchcd: int
    shrcd: int
    adj_prc: Optional[float] = None
    market_cap: Optional[float] = None

    model_config = {"strict": False}

    @field_validator("prc")
    @classmethod
    def prc_not_zero(cls, v: float) -> float:
        if v == 0.0:
            raise ValueError("prc must not be zero")
        return v

    @field_validator("permno")
    @classmethod
    def permno_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("permno must be positive")
        return v


def _validate_schema(df: pd.DataFrame, sample_n: int = 500) -> None:
    """Validate a random sample of rows against the pydantic model."""
    n = min(sample_n, len(df))
    sample = df.sample(n=n, random_state=42)
    errors: list[str] = []
    for idx, row in sample.iterrows():
        try:
            CRSPRowSchema(**row.to_dict())
        except Exception as exc:
            errors.append(f"Row {idx}: {exc}")
    if errors:
        logger.warning(f"Schema validation found {len(errors)} issues in sample")
        for e in errors[:5]:
            logger.warning(e)
    else:
        logger.info(f"Schema validation passed on {n}-row sample")


# ---------------------------------------------------------------------------
# SQL query builder
# ---------------------------------------------------------------------------


def _build_query(year: int) -> str:
    """Build the SQL query for a single calendar year."""
    cols = ", ".join(_CRSP_COLUMNS)
    return f"""
        SELECT {cols}
        FROM crsp.dsf AS a
        INNER JOIN crsp.dsenames AS b
            ON a.permno = b.permno
           AND a.date >= b.namedt
           AND a.date <= b.nameendt
        WHERE a.date BETWEEN '{year}-01-01' AND '{year}-12-31'
        ORDER BY a.permno, a.date
    """


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalization steps to raw CRSP data.

    Steps
    -----
    1. Convert ``date`` to datetime.
    2. Cast numeric columns to float64.
    3. Filter: ``shrcd`` in (10, 11), ``exchcd`` in (1, 2, 3).
    4. Remove rows where ``prc`` is null or zero.
    5. Set impossible returns (|ret| > 3.0) to NaN.
    6. Compute ``adj_prc = |prc| / cfacpr``.
    7. Compute ``market_cap = |prc| * shrout * 1000``.
    8. Remove duplicates on (permno, date).
    9. Sort by permno, date.
    """
    logger.debug(f"Normalizing {len(df):,} rows")

    # 1. Date conversion
    df["date"] = pd.to_datetime(df["date"])

    # 2. Cast numeric columns
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # 3. Universe filter: common stocks on major exchanges
    df = df[
        df["shrcd"].isin(COMMON_STOCK_SHRCDS) & df["exchcd"].isin(MAJOR_EXCHANGES)
    ].copy()
    logger.debug(f"  After shrcd/exchcd filter: {len(df):,}")

    # 4. Drop rows with null or zero price
    df = df[df["prc"].notna() & (df["prc"] != 0.0)].copy()
    logger.debug(f"  After prc filter: {len(df):,}")

    # 5. Cap impossible returns
    ret_mask = df["ret"].abs() > _MAX_RETURN
    n_capped = ret_mask.sum()
    if n_capped > 0:
        logger.info(f"  Setting {n_capped:,} impossible returns to NaN")
        df.loc[ret_mask, "ret"] = np.nan

    retx_mask = df["retx"].abs() > _MAX_RETURN
    df.loc[retx_mask, "retx"] = np.nan

    # 6. Adjusted price
    df["adj_prc"] = df["prc"].abs() / df["cfacpr"].replace(0, np.nan)

    # 7. Market capitalisation (shrout is in thousands of shares on WRDS)
    df["market_cap"] = df["prc"].abs() * df["shrout"] * 1_000.0

    # 8. De-duplicate
    before = len(df)
    df = df.drop_duplicates(subset=["permno", "date"], keep="last")
    dupes = before - len(df)
    if dupes > 0:
        logger.info(f"  Removed {dupes:,} duplicate (permno, date) rows")

    # 9. Sort
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

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
    """Pull CRSP daily data from WRDS, normalize, and save to parquet.

    Parameters
    ----------
    start_year : int
        First calendar year to pull (default from ``constants.START_DATE``).
    end_year : int, optional
        Last calendar year (inclusive). Defaults to the current year.
    output_dir : Path, optional
        Override the default output directory.

    Returns
    -------
    pd.DataFrame
        The concatenated, normalised CRSP dataset.
    """
    setup_logger()
    if end_year is None:
        end_year = dt.date.today().year
    if output_dir is None:
        output_dir = _OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pulling CRSP data for {start_year}–{end_year}")

    conn = get_wrds_connection()
    years = list(range(start_year, end_year + 1))
    chunks: list[pd.DataFrame] = []

    for year in tqdm(years, desc="CRSP yearly pulls"):
        cache_file = output_dir / f"year={year}" / "part-0.parquet"
        if cache_file.exists():
            logger.info(f"  {year}: loading from cache")
            chunk = pd.read_parquet(cache_file)
        else:
            query = _build_query(year)
            logger.info(f"  {year}: querying WRDS …")
            raw = conn.query(query, date_cols=["date"])

            if raw.empty:
                logger.warning(f"  {year}: no data returned, skipping")
                continue

            chunk = _normalize(raw)

            # Validate schema on the first chunk and then every 5 years
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
        logger.error("No data pulled from CRSP")
        return pd.DataFrame()

    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values(["permno", "date"]).reset_index(drop=True)
    logger.success(f"CRSP pull complete: {len(result):,} total rows")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pull()
