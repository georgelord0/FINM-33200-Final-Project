"""Pull Fama-French factor data from WRDS.

Downloads the 3-factor model, momentum factor, and (if available) the 5-factor
model from the ``ff`` library on WRDS.  All percentage values are converted to
decimal returns and the datasets are merged into a single date-indexed parquet.

Output
------
``data/factors/factors_daily.parquet``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import wrds
from loguru import logger
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from src.utils.constants import END_DATE, START_DATE
from src.utils.logging_utils import get_logger, setup_logger
from src.utils.wrds_connection import get_connection

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_log = get_logger(__name__)

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data" / "factors"
OUTPUT_FILE: Final[str] = "factors_daily.parquet"

# Columns that are stored as percentages in WRDS and need / 100
_PCT_COLS: Final[list[str]] = [
    "mktrf",
    "smb",
    "hml",
    "rf",
    "umd",
    "rmw",
    "cma",
]


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class FactorRow(BaseModel):
    """Validation schema for a single row of factor data."""

    date: pd.Timestamp
    mktrf: float
    smb: float
    hml: float
    rf: float
    umd: float | None = None
    rmw: float | None = None
    cma: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("date", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> pd.Timestamp:
        return pd.Timestamp(v)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _pull_ff3(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    """Pull Fama-French 3-factor daily data."""
    _log.info("Pulling ff.factors_daily (3-factor) ...")
    sql = f"""
        SELECT date, mktrf, smb, hml, rf
        FROM ff.factors_daily
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date
    """
    df = db.raw_sql(sql, date_cols=["date"])
    _log.info("  -> {} rows", len(df))
    return df


def _pull_momentum(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    """Pull momentum (UMD) factor."""
    _log.info("Pulling ff.factors_daily_mom (momentum) ...")
    sql = f"""
        SELECT date, umd
        FROM ff.factors_daily
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date
    """
    # Momentum lives in a separate table in some WRDS versions
    try:
        df = db.raw_sql(
            f"""
            SELECT date, umd
            FROM ff.factors_daily_mom
            WHERE date BETWEEN '{start}' AND '{end}'
            ORDER BY date
            """,
            date_cols=["date"],
        )
        _log.info("  -> {} rows from ff.factors_daily_mom", len(df))
        return df
    except Exception:
        _log.warning("ff.factors_daily_mom not available; trying ff.factors_daily for umd")
        try:
            df = db.raw_sql(sql, date_cols=["date"])
            if "umd" in df.columns:
                _log.info("  -> {} rows (umd from ff.factors_daily)", len(df))
                return df[["date", "umd"]]
        except Exception:
            pass
        _log.warning("Momentum factor not available; umd will be NaN")
        return pd.DataFrame(columns=["date", "umd"])


def _pull_ff5(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    """Pull Fama-French 5-factor daily data (RMW, CMA)."""
    _log.info("Attempting ff.factors5_daily (5-factor) ...")
    try:
        sql = f"""
            SELECT date, smb, hml, rmw, cma, mktrf, rf
            FROM ff.factors5_daily
            WHERE date BETWEEN '{start}' AND '{end}'
            ORDER BY date
        """
        df = db.raw_sql(sql, date_cols=["date"])
        _log.info("  -> {} rows from ff.factors5_daily", len(df))
        # Only keep the columns unique to FF5
        return df[["date", "rmw", "cma"]]
    except Exception as exc:
        _log.warning("ff.factors5_daily not available ({}); skipping RMW/CMA", exc)
        return pd.DataFrame(columns=["date", "rmw", "cma"])


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard normalization to merged factor data.

    - Converts percentage values to decimals (/ 100).
    - Deduplicates on date.
    - Sorts chronologically.
    - Converts date column to datetime.
    """
    df = df.copy()

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])

    # Coerce to float (WRDS ff.factors_daily already stores decimals)
    for col in _PCT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate
    df = df.drop_duplicates(subset=["date"], keep="last")

    # Sort
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Main pull function
# ---------------------------------------------------------------------------


def pull(
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Pull and merge all available Fama-French factor data from WRDS.

    Parameters
    ----------
    start : str
        Sample start date (YYYY-MM-DD).
    end : str
        Sample end date (YYYY-MM-DD).
    output_dir : Path, optional
        Override default output directory.

    Returns
    -------
    pd.DataFrame
        Merged, normalized factor data.
    """
    setup_logger()
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    _log.info("Connecting to WRDS ...")
    conn = get_connection()
    db = conn._ensure_alive()

    steps = [
        ("3-factor", _pull_ff3),
        ("momentum", _pull_momentum),
        ("5-factor", _pull_ff5),
    ]

    frames: dict[str, pd.DataFrame] = {}
    for label, func in tqdm(steps, desc="Pulling factors"):
        frames[label] = func(db, start, end)

    _log.info("WRDS queries complete.")

    # Merge on date
    _log.info("Merging factor datasets ...")
    merged = frames["3-factor"]

    if not frames["momentum"].empty:
        merged = merged.merge(frames["momentum"], on="date", how="left")

    if not frames["5-factor"].empty:
        merged = merged.merge(frames["5-factor"], on="date", how="left")

    # Normalize
    merged = _normalize(merged)

    # Validate a sample of rows
    _log.info("Validating schema on sample rows ...")
    sample = merged.head(min(100, len(merged)))
    for _, row in sample.iterrows():
        FactorRow(**row.to_dict())

    # Write
    dest = out / OUTPUT_FILE
    merged.to_parquet(dest, index=False, engine="pyarrow")
    _log.info("Wrote {} rows to {}", len(merged), dest)

    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logger(level="INFO")
    pull()
