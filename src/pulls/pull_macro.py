"""Pull macroeconomic series from the WRDS FRED mirror.

Retrieves key macro indicators (Fed Funds rate, Treasury yields, CPI,
unemployment, industrial production, VIX) from ``wrdsapps.fred_data``
or a compatible WRDS-hosted FRED table.  Data is pivoted to wide format,
forward-filled, and stored as a single parquet file.

Output
------
``data/macro/macro_daily.parquet``
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

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data" / "macro"
OUTPUT_FILE: Final[str] = "macro_daily.parquet"

# FRED series IDs to pull
SERIES_IDS: Final[list[str]] = [
    "FEDFUNDS",
    "GS10",
    "TB3MS",
    "CPIAUCSL",
    "UNRATE",
    "INDPRO",
    "VIXCLS",
]

# Mapping from FRED series ID -> standardized column name
COLUMN_MAP: Final[dict[str, str]] = {
    "FEDFUNDS": "fed_funds",
    "GS10": "gs10",
    "TB3MS": "tb3ms",
    "CPIAUCSL": "cpi",
    "UNRATE": "unrate",
    "INDPRO": "indpro",
    "VIXCLS": "vix",
}

# Known WRDS table names for FRED data (tried in order)
_FRED_TABLES: Final[list[str]] = [
    "wrdsapps.fred_data",
    "wrdsapps.fredmd",
    "wrdsapps.fred_series",
]


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class MacroRow(BaseModel):
    """Validation schema for a single row of macro data."""

    date: pd.Timestamp
    fed_funds: float | None = None
    gs10: float | None = None
    tb3ms: float | None = None
    cpi: float | None = None
    unrate: float | None = None
    indpro: float | None = None
    vix: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("date", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> pd.Timestamp:
        return pd.Timestamp(v)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _try_pull_fred(
    db: wrds.Connection,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Attempt to pull FRED data from known WRDS mirror tables.

    Returns the raw long-format DataFrame on success, or ``None`` if no
    table is accessible.
    """
    series_str = ", ".join(f"'{s}'" for s in SERIES_IDS)

    for table in tqdm(_FRED_TABLES, desc="Trying FRED tables"):
        try:
            _log.info("Trying {} ...", table)
            sql = f"""
                SELECT date, series_id, value
                FROM {table}
                WHERE series_id IN ({series_str})
                  AND date BETWEEN '{start}' AND '{end}'
                ORDER BY date
            """
            df = db.raw_sql(sql, date_cols=["date"])
            if df is not None and not df.empty:
                _log.info("  -> {} rows from {}", len(df), table)
                return df
        except Exception as exc:
            _log.debug("  {} failed: {}", table, exc)
            continue

    return None


def _try_pull_fred_api(start: str, end: str) -> pd.DataFrame | None:
    """Fallback: pull FRED data via pandas_datareader when WRDS mirror is unavailable."""
    try:
        import pandas_datareader.data as web

        _log.info("Falling back to pandas_datareader FRED API ...")
        frames = []
        for series_id in tqdm(SERIES_IDS, desc="Pulling FRED series"):
            try:
                s = web.DataReader(series_id, "fred", start, end)
                s = s.reset_index().melt(id_vars="DATE", var_name="series_id", value_name="value")
                s = s.rename(columns={"DATE": "date"})
                frames.append(s)
            except Exception as exc:
                _log.warning("  Failed to pull {}: {}", series_id, exc)
        if frames:
            result = pd.concat(frames, ignore_index=True)
            _log.info("  -> {} total rows from FRED API", len(result))
            return result
    except ImportError:
        _log.warning("pandas_datareader not installed; cannot use FRED API fallback")
    return None


def _create_empty_macro() -> pd.DataFrame:
    """Create an empty DataFrame with the correct schema."""
    cols = {"date": pd.Series(dtype="datetime64[ns]")}
    for std_name in COLUMN_MAP.values():
        cols[std_name] = pd.Series(dtype="float64")
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long -> wide, rename, ffill, sort.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format FRED data with columns ``date``, ``series_id``, ``value``.

    Returns
    -------
    pd.DataFrame
        Wide-format, forward-filled, with standardized column names.
    """
    df = df.copy()

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])

    # Coerce value
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Deduplicate: keep last occurrence per date/series
    df = df.drop_duplicates(subset=["date", "series_id"], keep="last")

    # Pivot to wide: date x series_id
    wide = df.pivot(index="date", columns="series_id", values="value")

    # Rename columns to standardized lowercase names
    wide = wide.rename(columns=COLUMN_MAP)

    # Ensure all expected columns exist
    for std_name in COLUMN_MAP.values():
        if std_name not in wide.columns:
            wide[std_name] = pd.NA

    # Keep only standardized columns
    wide = wide[list(COLUMN_MAP.values())]

    # Sort chronologically
    wide = wide.sort_index()

    # Forward-fill (macro data has mixed frequencies)
    wide = wide.ffill()

    # Reset index to have date as a regular column
    wide = wide.reset_index()
    wide = wide.rename(columns={"index": "date"} if "date" not in wide.columns else {})

    return wide


# ---------------------------------------------------------------------------
# Main pull function
# ---------------------------------------------------------------------------


def pull(
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Pull macroeconomic data from the WRDS FRED mirror.

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
        Wide-format macro data with standardized column names.
    """
    setup_logger()
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    _log.info("Connecting to WRDS ...")
    conn = get_connection()
    db = conn._ensure_alive()

    raw = _try_pull_fred(db, start, end)
    _log.info("WRDS queries complete.")

    if raw is None or raw.empty:
        _log.warning("WRDS FRED mirror not available; trying FRED API fallback ...")
        raw = _try_pull_fred_api(start, end)

    if raw is None or raw.empty:
        _log.warning(
            "No FRED data from any source. "
            "Writing empty placeholder parquet with correct schema."
        )
        macro = _create_empty_macro()
    else:
        macro = _normalize(raw)

    # Validate a sample of rows
    if not macro.empty:
        _log.info("Validating schema on sample rows ...")
        sample = macro.head(min(100, len(macro)))
        for _, row in sample.iterrows():
            MacroRow(**row.to_dict())

    # Write
    dest = out / OUTPUT_FILE
    macro.to_parquet(dest, index=False, engine="pyarrow")
    _log.info("Wrote {} rows to {}", len(macro), dest)

    return macro


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logger(level="INFO")
    pull()
