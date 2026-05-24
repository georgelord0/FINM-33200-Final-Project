"""Pull and normalize daily risk-free rate and Fama-French factors from WRDS.

Downloads ff.factors_daily, converts from percentages to decimals,
and stores as a single parquet file under data/riskfree/.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator

from src.utils.constants import START_DATE
from src.utils.wrds_connection import get_connection as get_wrds_connection
from src.utils.logging_utils import get_logger, setup_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "riskfree"
_OUTPUT_FILE = _OUTPUT_DIR / "ff_factors_daily.parquet"

_FACTOR_COLS: list[str] = ["rf", "mktrf", "smb", "hml"]
_PERCENTAGE_DIVISOR: float = 100.0


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class RiskFreeRowSchema(BaseModel):
    """Validates a single row of the normalised factor dataset."""

    date: dt.date
    rf: Optional[float] = None
    mktrf: Optional[float] = None
    smb: Optional[float] = None
    hml: Optional[float] = None

    model_config = {"strict": False}


def _validate_schema(df: pd.DataFrame, sample_n: int = 500) -> None:
    """Validate a random sample of rows against the pydantic model."""
    n = min(sample_n, len(df))
    sample = df.sample(n=n, random_state=42)
    errors: list[str] = []
    for idx, row in sample.iterrows():
        try:
            RiskFreeRowSchema(**row.to_dict())
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


def _build_query(start_date: str) -> str:
    """Build SQL query for Fama-French daily factors."""
    return f"""
        SELECT date, rf, mktrf, smb, hml
        FROM ff.factors_daily
        WHERE date >= '{start_date}'
        ORDER BY date
    """


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Fama-French factor data.

    Steps
    -----
    1. Convert ``date`` to datetime.
    2. Divide all factor values by 100 (WRDS stores as percentages).
    3. Deduplicate on ``date``.
    4. Sort chronologically.
    """
    logger.debug(f"Normalizing {len(df):,} factor rows")

    # 1. Date conversion
    df["date"] = pd.to_datetime(df["date"])

    # 2. Coerce to float (WRDS ff.factors_daily already stores decimals)
    for col in _FACTOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    logger.info(
        f"  Factor summary: "
        f"rf mean={df['rf'].mean():.6f}, mktrf mean={df['mktrf'].mean():.6f}"
    )

    # 3. Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["date"], keep="last")
    dupes = before - len(df)
    if dupes > 0:
        logger.info(f"  Removed {dupes:,} duplicate date rows")

    # 4. Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    logger.debug(f"  Final normalised row count: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Main pull function
# ---------------------------------------------------------------------------


def pull(
    start_date: str = START_DATE,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Pull daily Fama-French factors from WRDS, normalize, and save.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format (default from constants).
    output_dir : Path, optional
        Override the default output directory.

    Returns
    -------
    pd.DataFrame
        The normalised daily factor dataset.
    """
    setup_logger()
    if output_dir is None:
        output_dir = _OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ff_factors_daily.parquet"

    if output_file.exists():
        logger.info(f"Loading cached factor data from {output_file}")
        return pd.read_parquet(output_file)

    logger.info(f"Pulling Fama-French daily factors from {start_date}")
    conn = get_wrds_connection()
    query = _build_query(start_date)
    raw = conn.query(query, date_cols=["date"])
    conn.close()

    if raw.empty:
        logger.error("No data returned for FF factors")
        return pd.DataFrame()

    result = _normalize(raw)
    _validate_schema(result)

    result.to_parquet(output_file, index=False, engine="pyarrow")
    logger.success(
        f"FF factors saved: {len(result):,} rows "
        f"({result['date'].min().date()} → {result['date'].max().date()}) "
        f"→ {output_file}"
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pull()
