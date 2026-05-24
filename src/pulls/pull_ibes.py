"""Pull analyst estimate and recommendation data from I/B/E/S on WRDS.

Retrieves EPS summary statistics from ``ibes.statsum_epsus`` and analyst
recommendations from ``ibes.recddet``.  Computes an earnings surprise
measure and standardizes column names to snake_case.

Output
------
``data/ibes/ibes_estimates.parquet``
``data/ibes/ibes_recommendations.parquet``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data" / "ibes"
ESTIMATES_FILE: Final[str] = "ibes_estimates.parquet"
RECOMMENDATIONS_FILE: Final[str] = "ibes_recommendations.parquet"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class EstimateRow(BaseModel):
    """Validation schema for a single row of EPS estimate data."""

    ticker: str
    stat_period: pd.Timestamp
    fp_end_date: pd.Timestamp
    mean_est: float | None = None
    med_est: float | None = None
    std_dev: float | None = None
    num_est: int | None = None
    actual: float | None = None
    surprise: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("stat_period", "fp_end_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> pd.Timestamp:
        return pd.Timestamp(v)


class RecommendationRow(BaseModel):
    """Validation schema for a single row of recommendation data."""

    ticker: str
    announce_date: pd.Timestamp
    recommendation: int | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("announce_date", mode="before")
    @classmethod
    def _coerce_date(cls, v: object) -> pd.Timestamp:
        return pd.Timestamp(v)


# Arrow schemas for empty-file fallback
_ESTIMATES_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        ("ticker", pa.string()),
        ("stat_period", pa.timestamp("ns")),
        ("fp_end_date", pa.timestamp("ns")),
        ("mean_est", pa.float64()),
        ("med_est", pa.float64()),
        ("std_dev", pa.float64()),
        ("num_est", pa.int64()),
        ("actual", pa.float64()),
        ("surprise", pa.float64()),
    ]
)

_RECOMMENDATIONS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        ("ticker", pa.string()),
        ("announce_date", pa.timestamp("ns")),
        ("recommendation", pa.int64()),
    ]
)


# ---------------------------------------------------------------------------
# Column rename maps (WRDS -> snake_case)
# ---------------------------------------------------------------------------

_ESTIMATES_RENAME: Final[dict[str, str]] = {
    "ticker": "ticker",
    "statpers": "stat_period",
    "fpedats": "fp_end_date",
    "meanest": "mean_est",
    "medest": "med_est",
    "stdev": "std_dev",
    "numest": "num_est",
    "actual": "actual",
}

_RECOMMENDATIONS_RENAME: Final[dict[str, str]] = {
    "ticker": "ticker",
    "anndats": "announce_date",
    "ireccd": "recommendation",
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def _pull_estimates(
    db: wrds.Connection,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Pull EPS summary statistics from ibes.statsum_epsus."""
    _log.info("Pulling ibes.statsum_epsus ...")
    try:
        sql = f"""
            SELECT ticker, statpers, fpedats,
                   meanest, medest, stdev, numest, actual
            FROM ibes.statsum_epsus
            WHERE statpers BETWEEN '{start}' AND '{end}'
              AND fpi = '1'
            ORDER BY ticker, statpers
        """
        df = db.raw_sql(sql, date_cols=["statpers", "fpedats"])
        _log.info("  -> {} rows", len(df))
        return df
    except Exception as exc:
        _log.warning("Failed to pull ibes.statsum_epsus: {}", exc)
        return None


def _pull_recommendations(
    db: wrds.Connection,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Pull analyst recommendations from ibes.recddet."""
    _log.info("Pulling ibes.recddet ...")
    try:
        sql = f"""
            SELECT ticker, anndats, ireccd
            FROM ibes.recddet
            WHERE anndats BETWEEN '{start}' AND '{end}'
            ORDER BY ticker, anndats
        """
        df = db.raw_sql(sql, date_cols=["anndats"])
        _log.info("  -> {} rows", len(df))
        return df
    except Exception as exc:
        _log.warning("Failed to pull ibes.recddet: {}", exc)
        return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _normalize_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize EPS estimate data.

    - Renames columns to snake_case.
    - Converts dates to datetime.
    - Deduplicates on (ticker, stat_period, fp_end_date).
    - Computes surprise = (actual - mean_est) / std_dev.
    - Sorts by ticker, stat_period.
    """
    df = df.copy()

    # Rename
    df = df.rename(columns=_ESTIMATES_RENAME)

    # Ensure datetime
    df["stat_period"] = pd.to_datetime(df["stat_period"])
    df["fp_end_date"] = pd.to_datetime(df["fp_end_date"])

    # Coerce numerics
    for col in ("mean_est", "med_est", "std_dev", "num_est", "actual"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate
    df = df.drop_duplicates(
        subset=["ticker", "stat_period", "fp_end_date"], keep="last"
    )

    # Compute earnings surprise: (actual - mean_est) / std_dev
    mask = df["std_dev"].notna() & (df["std_dev"] != 0) & df["actual"].notna() & df["mean_est"].notna()
    df["surprise"] = np.where(
        mask,
        (df["actual"] - df["mean_est"]) / df["std_dev"],
        np.nan,
    )

    # Sort
    df = df.sort_values(["ticker", "stat_period"]).reset_index(drop=True)

    return df


def _normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize recommendation data.

    - Renames columns to snake_case.
    - Converts dates to datetime.
    - Deduplicates on (ticker, announce_date).
    - Sorts by ticker, announce_date.
    """
    df = df.copy()

    # Rename
    df = df.rename(columns=_RECOMMENDATIONS_RENAME)

    # Ensure datetime
    df["announce_date"] = pd.to_datetime(df["announce_date"])

    # Coerce recommendation code to numeric
    df["recommendation"] = pd.to_numeric(df["recommendation"], errors="coerce")

    # Deduplicate
    df = df.drop_duplicates(subset=["ticker", "announce_date"], keep="last")

    # Sort
    df = df.sort_values(["ticker", "announce_date"]).reset_index(drop=True)

    return df


def _write_empty_parquet(dest: Path, schema: pa.Schema) -> None:
    """Write an empty parquet file with the specified schema."""
    table = pa.table(
        {field.name: pa.array([], type=field.type) for field in schema},
        schema=schema,
    )
    pq.write_table(table, dest)


# ---------------------------------------------------------------------------
# Main pull function
# ---------------------------------------------------------------------------


def pull(
    start: str = START_DATE,
    end: str = END_DATE,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull I/B/E/S estimates and recommendations from WRDS.

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
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (estimates, recommendations) DataFrames.
    """
    setup_logger()
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    _log.info("Connecting to WRDS ...")
    try:
        conn = get_connection()
        db = conn._ensure_alive()
    except Exception as exc:
        _log.error("Failed to connect to WRDS: {}", exc)
        _log.warning("Writing empty parquet files with correct schemas.")
        _write_empty_parquet(out / ESTIMATES_FILE, _ESTIMATES_SCHEMA)
        _write_empty_parquet(out / RECOMMENDATIONS_FILE, _RECOMMENDATIONS_SCHEMA)
        est_empty = pd.DataFrame(columns=[f.name for f in _ESTIMATES_SCHEMA])
        rec_empty = pd.DataFrame(columns=[f.name for f in _RECOMMENDATIONS_SCHEMA])
        return est_empty, rec_empty

    steps = [
        ("estimates", _pull_estimates),
        ("recommendations", _pull_recommendations),
    ]

    raw: dict[str, pd.DataFrame | None] = {}
    for label, func in tqdm(steps, desc="Pulling I/B/E/S"):
        raw[label] = func(db, start, end)

    _log.info("WRDS queries complete.")

    # --- Estimates ---
    if raw["estimates"] is not None and not raw["estimates"].empty:
        estimates = _normalize_estimates(raw["estimates"])

        # Validate sample
        _log.info("Validating estimate schema on sample rows ...")
        for _, row in estimates.head(min(100, len(estimates))).iterrows():
            EstimateRow(**row.to_dict())
    else:
        _log.warning("I/B/E/S estimates unavailable; writing empty placeholder.")
        estimates = pd.DataFrame(columns=[f.name for f in _ESTIMATES_SCHEMA])

    est_dest = out / ESTIMATES_FILE
    if estimates.empty:
        _write_empty_parquet(est_dest, _ESTIMATES_SCHEMA)
    else:
        estimates.to_parquet(est_dest, index=False, engine="pyarrow")
    _log.info("Wrote {} estimate rows to {}", len(estimates), est_dest)

    # --- Recommendations ---
    if raw["recommendations"] is not None and not raw["recommendations"].empty:
        recommendations = _normalize_recommendations(raw["recommendations"])

        # Validate sample
        _log.info("Validating recommendation schema on sample rows ...")
        for _, row in recommendations.head(min(100, len(recommendations))).iterrows():
            RecommendationRow(**row.to_dict())
    else:
        _log.warning("I/B/E/S recommendations unavailable; writing empty placeholder.")
        recommendations = pd.DataFrame(
            columns=[f.name for f in _RECOMMENDATIONS_SCHEMA]
        )

    rec_dest = out / RECOMMENDATIONS_FILE
    if recommendations.empty:
        _write_empty_parquet(rec_dest, _RECOMMENDATIONS_SCHEMA)
    else:
        recommendations.to_parquet(rec_dest, index=False, engine="pyarrow")
    _log.info("Wrote {} recommendation rows to {}", len(recommendations), rec_dest)

    return estimates, recommendations


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logger(level="INFO")
    pull()
