"""Build the canonical panel dataset for the TSFM finance pipeline.

Loads CRSP returns, risk-free rates, and all engineered features from their
respective parquet directories, merges them into a single panel keyed by
(date, permno), enforces leakage-prevention rules, and persists the result
as ``data/datasets/panel.parquet`` partitioned by year.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.utils.constants import (
    COL_DATE,
    COL_MARKET_CAP,
    COL_PERMNO,
    COL_PRC,
    COL_RET,
    COL_RF,
    COL_SHROUT,
    MIN_PRICE,
    TRADING_DAYS_PER_QUARTER,
)
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# SIC-code to sector mapping
# ---------------------------------------------------------------------------

_SIC_SECTOR_MAP: dict[tuple[int, int], str] = {
    (100, 999): "Agriculture",
    (1000, 1499): "Mining",
    (1500, 1799): "Construction",
    (2000, 3999): "Manufacturing",
    (4000, 4999): "Transportation & Utilities",
    (5000, 5199): "Wholesale Trade",
    (5200, 5999): "Retail Trade",
    (6000, 6799): "Finance",
    (7000, 8999): "Services",
    (9100, 9729): "Public Administration",
}


def _siccd_to_sector(siccd: pd.Series) -> pd.Series:
    """Map SIC codes to broad sector labels.

    Parameters
    ----------
    siccd:
        Series of integer SIC codes.

    Returns
    -------
    pd.Series
        Sector labels; unmapped codes become ``"Other"``.
    """

    def _map_one(code: float | int) -> str:
        if pd.isna(code):
            return "Other"
        code_int = int(code)
        for (lo, hi), sector in _SIC_SECTOR_MAP.items():
            if lo <= code_int <= hi:
                return sector
        return "Other"

    return siccd.apply(_map_one)


# ---------------------------------------------------------------------------
# Fundamental forward-fill helpers
# ---------------------------------------------------------------------------

_FUNDAMENTAL_PREFIXES: tuple[str, ...] = (
    "book_to_market",
    "roe",
    "asset_growth",
    "gross_profitability",
    "at",
    "lt",
    "seq",
    "ceqq",
    "revtq",
    "niq",
    "ibq",
    "oiadpq",
)

_MAX_FFILL_GAP: int = TRADING_DAYS_PER_QUARTER * 2  # 126 trading days


def _is_fundamental_col(col: str) -> bool:
    """Return True if *col* is an accounting / fundamental feature."""
    return any(col.startswith(p) or col == p for p in _FUNDAMENTAL_PREFIXES)


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def _load_config(config_path: str | None) -> dict:
    """Load a YAML config or return sensible defaults.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.  If ``None``, paths are resolved
        relative to the project root using ``src/configs/paths.yaml``.

    Returns
    -------
    dict
        Configuration dictionary with at least ``features_dir``,
        ``crsp_dir``, ``riskfree_dir``, and ``datasets_dir`` keys.
    """
    root = _project_root()

    def _resolve_paths(cfg: dict, base: Path) -> dict:
        for key in ("features_dir", "crsp_dir", "riskfree_dir", "datasets_dir"):
            if key in cfg and not Path(cfg[key]).is_absolute():
                cfg[key] = str(base / cfg[key])
        return cfg

    if config_path is not None:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
        # Explicit configs resolve against project root.
        return _resolve_paths(cfg, root)

    # Default: load from paths.yaml
    paths_yaml = root / "src" / "configs" / "paths.yaml"
    if paths_yaml.exists():
        with open(paths_yaml) as fh:
            cfg = yaml.safe_load(fh)
        # The src pipeline writes under src/data by default; keep the default
        # panel builder aligned with the pull and feature modules.
        return _resolve_paths(cfg, root / "src")

    # Fallback hard-coded defaults
    return {
        "features_dir": str(root / "data" / "features"),
        "crsp_dir": str(root / "data" / "crsp"),
        "riskfree_dir": str(root / "data" / "riskfree"),
        "datasets_dir": str(root / "data" / "datasets"),
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_parquets(directory: Path, label: str) -> pd.DataFrame:
    """Read parquet files from a flat or partitioned directory.

    Parameters
    ----------
    directory:
        Directory containing one or more parquet files.
    label:
        Human-readable label used in log messages.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all parquet files found.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist or contains no parquet files.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")

    files = sorted(directory.glob("*.parquet"))
    if files:
        frames: list[pd.DataFrame] = []
        for f in tqdm(files, desc=f"Loading {label}", leave=False):
            frames.append(pd.read_parquet(f))

        df = pd.concat(frames, ignore_index=True)
        log.info("Loaded {} rows from {} ({} files)", len(df), label, len(files))
        return df

    partition_files = sorted(directory.rglob("*.parquet"))
    if partition_files:
        df = pd.read_parquet(directory)
        log.info(
            "Loaded {} rows from {} partitioned parquet files in {}",
            len(df),
            len(partition_files),
            label,
        )
        return df

    if not partition_files:
        raise FileNotFoundError(f"No parquet files found in {directory} ({label})")

    raise AssertionError("unreachable")


_FEATURE_FILE_EXCLUDES = {"targets.parquet"}
_UNSAFE_FEATURE_COLUMNS = {
    COL_RET,
    "retx",
    COL_PRC,
    "adj_prc",
    "vol",
    COL_SHROUT,
    "cfacpr",
    "cfacshr",
    "siccd",
    "exchcd",
    "shrcd",
    COL_MARKET_CAP,
    "target_ret",
    "excess_ret",
    "direction",
    "year",
}


def _load_feature_files(features_dir: Path) -> pd.DataFrame:
    """Load engineered features and merge feature families by key.

    Feature files keyed by ``(date, permno)`` are merged horizontally. Macro
    files keyed only by ``date`` are merged afterward and broadcast by date.
    Files containing model targets/raw labels are excluded.
    """
    features_dir = Path(features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory does not exist: {features_dir}")

    files = sorted(
        p for p in features_dir.glob("*.parquet")
        if p.name not in _FEATURE_FILE_EXCLUDES
    )
    if not files:
        raise FileNotFoundError(f"No feature parquet files found in {features_dir}")

    asset_features: pd.DataFrame | None = None
    date_features: list[pd.DataFrame] = []

    for path in tqdm(files, desc="Loading Features", leave=False):
        df = pd.read_parquet(path)
        if df.empty:
            log.warning("Skipping empty feature file {}", path.name)
            continue
        if COL_DATE not in df.columns:
            log.warning("Skipping feature file without date column: {}", path.name)
            continue

        df[COL_DATE] = pd.to_datetime(df[COL_DATE])
        keys = [COL_DATE]
        if COL_PERMNO in df.columns:
            df[COL_PERMNO] = df[COL_PERMNO].astype(int)
            keys.append(COL_PERMNO)

        drop_cols = [
            c for c in df.columns
            if c not in keys and c in _UNSAFE_FEATURE_COLUMNS
        ]
        if drop_cols:
            log.warning("Dropping unsafe columns from {}: {}", path.name, drop_cols)
            df = df.drop(columns=drop_cols)

        df = df.sort_values(keys).drop_duplicates(keys, keep="last")

        if COL_PERMNO in keys:
            if asset_features is None:
                asset_features = df
            else:
                asset_features = asset_features.merge(
                    df,
                    on=[COL_DATE, COL_PERMNO],
                    how="outer",
                    validate="one_to_one",
                )
        else:
            date_features.append(df)

        log.info("Loaded feature file {}: {} rows, {} columns", path.name, len(df), len(df.columns))

    if asset_features is None:
        raise ValueError(f"No asset-level feature files found in {features_dir}")

    for df in date_features:
        asset_features = asset_features.merge(df, on=COL_DATE, how="left", validate="many_to_one")

    log.info(
        "Merged engineered features: {} rows, {} columns",
        len(asset_features),
        len(asset_features.columns),
    )
    return asset_features


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_panel(config_path: str | None = None) -> pd.DataFrame:
    """Build the canonical panel dataset.

    The panel contains one row per (date, permno) observation with the
    forward-looking target return, excess return, market cap, sector, and
    all engineered features.  Leakage-prevention assertions are enforced
    before saving.

    Parameters
    ----------
    config_path:
        Optional path to a YAML config file that specifies data directories.
        When ``None``, directories are resolved from
        ``src/configs/paths.yaml`` or project-root defaults.

    Returns
    -------
    pd.DataFrame
        The merged panel dataset.
    """
    cfg = _load_config(config_path)
    features_dir = Path(cfg["features_dir"])
    crsp_dir = Path(cfg["crsp_dir"])
    riskfree_dir = Path(cfg["riskfree_dir"])
    datasets_dir = Path(cfg["datasets_dir"])

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    log.info("Loading CRSP data ...")
    crsp = _load_parquets(crsp_dir, "CRSP")
    crsp[COL_DATE] = pd.to_datetime(crsp[COL_DATE])
    crsp[COL_PERMNO] = crsp[COL_PERMNO].astype(int)

    log.info("Loading risk-free rate data ...")
    rf = _load_parquets(riskfree_dir, "Risk-free")
    rf[COL_DATE] = pd.to_datetime(rf[COL_DATE])

    log.info("Loading engineered features ...")
    features = _load_feature_files(features_dir)

    # ------------------------------------------------------------------
    # 2. Compute market cap (if not already present)
    # ------------------------------------------------------------------
    if COL_MARKET_CAP not in crsp.columns and COL_PRC in crsp.columns and COL_SHROUT in crsp.columns:
        crsp[COL_MARKET_CAP] = crsp[COL_PRC].abs() * crsp[COL_SHROUT] * 1_000
        log.info("Computed market_cap from prc * shrout")

    # ------------------------------------------------------------------
    # 3. Map SIC code to sector
    # ------------------------------------------------------------------
    if "siccd" in crsp.columns:
        crsp["sector"] = _siccd_to_sector(crsp["siccd"])
    elif "sector" not in crsp.columns:
        crsp["sector"] = "Unknown"

    # ------------------------------------------------------------------
    # 4. Filter penny stocks
    # ------------------------------------------------------------------
    if COL_PRC in crsp.columns:
        n_before = len(crsp)
        crsp = crsp[crsp[COL_PRC].abs() >= MIN_PRICE].copy()
        log.info(
            "Removed {} penny-stock rows (price < {})",
            n_before - len(crsp),
            MIN_PRICE,
        )

    # ------------------------------------------------------------------
    # 5. Merge CRSP with risk-free rate
    # ------------------------------------------------------------------
    rf_cols = [COL_DATE] + [c for c in rf.columns if c != COL_DATE]
    panel = crsp.merge(rf[rf_cols], on=COL_DATE, how="left")
    log.info("Panel after CRSP + RF merge: {} rows", len(panel))

    # ------------------------------------------------------------------
    # 6. Compute excess return
    # ------------------------------------------------------------------
    rf_col = COL_RF if COL_RF in panel.columns else None
    if rf_col is None:
        # Try common alternatives
        for alt in ("rf", "RF", "risk_free", "tbill"):
            if alt in panel.columns:
                rf_col = alt
                break

    if rf_col is not None and COL_RET in panel.columns:
        panel["excess_ret"] = panel[COL_RET] - panel[rf_col]
    elif COL_RET in panel.columns:
        log.warning("No risk-free column found; excess_ret = ret")
        panel["excess_ret"] = panel[COL_RET]
    else:
        log.warning("No return column found in CRSP data")

    # ------------------------------------------------------------------
    # 7. Merge with features
    # ------------------------------------------------------------------
    merge_keys = [COL_DATE, COL_PERMNO]
    panel = panel.merge(features, on=merge_keys, how="inner", suffixes=("", "_feat"))
    log.info("Panel after feature merge: {} rows", len(panel))

    # Drop any duplicate suffixed columns
    dup_cols = [c for c in panel.columns if c.endswith("_feat")]
    if dup_cols:
        panel.drop(columns=dup_cols, inplace=True)

    # ------------------------------------------------------------------
    # 8. Forward-fill fundamentals (per permno, max gap = 126 days)
    # ------------------------------------------------------------------
    fundamental_cols = [c for c in panel.columns if _is_fundamental_col(c)]
    if fundamental_cols:
        log.info("Forward-filling {} fundamental columns (max gap={})", len(fundamental_cols), _MAX_FFILL_GAP)
        panel.sort_values([COL_PERMNO, COL_DATE], inplace=True)
        for col in tqdm(fundamental_cols, desc="Forward-filling fundamentals", leave=False):
            panel[col] = panel.groupby(COL_PERMNO)[col].transform(
                lambda s: s.ffill(limit=_MAX_FFILL_GAP)
            )

    # ------------------------------------------------------------------
    # 9. Create forward-looking target return
    # ------------------------------------------------------------------
    panel.sort_values([COL_PERMNO, COL_DATE], inplace=True)
    panel["target_ret"] = panel.groupby(COL_PERMNO)[COL_RET].shift(-1)
    log.info(
        "Created target_ret via shift(-1); {} NaN targets",
        panel["target_ret"].isna().sum(),
    )

    # ------------------------------------------------------------------
    # 10. Leakage-prevention assertions
    # ------------------------------------------------------------------
    # target_ret is forward-looking by construction (shift(-1) above).
    # Verify: for each permno group, target_ret[t] == ret[t+1].
    def _target_is_shifted(g: pd.DataFrame) -> bool:
        if len(g) <= 1:
            return True
        actual = g["target_ret"].iloc[:-1].reset_index(drop=True)
        expected = g[COL_RET].iloc[1:].reset_index(drop=True)
        return bool(actual.equals(expected))

    _sample = panel.groupby(COL_PERMNO).head(10)
    _check = _sample.groupby(COL_PERMNO).apply(_target_is_shifted)
    assert _check.all(), "Leakage check failed: target_ret is not correctly shifted"
    log.info("Leakage assertion passed: target_ret is forward-looking")

    # All feature columns should be backward-looking (no shift applied here;
    # they are computed from lagged data in the feature pipeline).
    feature_cols_in_panel = [c for c in features.columns if c not in merge_keys]
    log.info(
        "Feature columns treated as backward-looking ({} cols): {}",
        len(feature_cols_in_panel),
        feature_cols_in_panel[:10],
    )

    # ------------------------------------------------------------------
    # 11. Drop rows where target is NaN
    # ------------------------------------------------------------------
    n_before = len(panel)
    panel.dropna(subset=["target_ret"], inplace=True)
    log.info("Dropped {} rows with NaN target_ret", n_before - len(panel))

    # ------------------------------------------------------------------
    # 12. Organise columns
    # ------------------------------------------------------------------
    id_cols = [COL_DATE, COL_PERMNO, "target_ret", "excess_ret"]
    if COL_MARKET_CAP in panel.columns:
        id_cols.append(COL_MARKET_CAP)
    if "sector" in panel.columns:
        id_cols.append("sector")

    other_cols = sorted(c for c in panel.columns if c not in id_cols)
    panel = panel[id_cols + other_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 13. Summary statistics
    # ------------------------------------------------------------------
    log.info("--- Panel Summary ---")
    log.info("Shape: {}", panel.shape)
    log.info(
        "Date range: {} to {}",
        panel[COL_DATE].min().strftime("%Y-%m-%d"),
        panel[COL_DATE].max().strftime("%Y-%m-%d"),
    )
    log.info("Unique permnos: {}", panel[COL_PERMNO].nunique())
    log.info("Unique dates: {}", panel[COL_DATE].nunique())
    if "sector" in panel.columns:
        log.info("Sector distribution:\n{}", panel["sector"].value_counts().to_string())
    log.info(
        "Target ret stats:\n{}",
        panel["target_ret"].describe().to_string(),
    )
    log.info(
        "Missing values per column (top 20):\n{}",
        panel.isnull().sum().sort_values(ascending=False).head(20).to_string(),
    )

    # ------------------------------------------------------------------
    # 14. Save to parquet, partitioned by year
    # ------------------------------------------------------------------
    datasets_dir.mkdir(parents=True, exist_ok=True)
    panel["year"] = panel[COL_DATE].dt.year

    out_dir = datasets_dir / "panel.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(panel["year"].unique())
    for yr in tqdm(years, desc="Writing partitions"):
        part_dir = out_dir / f"year={yr}"
        part_dir.mkdir(parents=True, exist_ok=True)
        panel[panel["year"] == yr].drop(columns=["year"]).to_parquet(
            part_dir / "part.parquet", index=False
        )

    panel.drop(columns=["year"], inplace=True)
    log.info("Saved panel dataset to {} ({} year partitions)", out_dir, len(years))

    return panel


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.utils.logging_utils import setup_logger

    parser = argparse.ArgumentParser(description="Build canonical panel dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional)",
    )
    args = parser.parse_args()

    setup_logger(level="INFO")
    df = build_panel(config_path=args.config)
    log.info("Done. Final panel shape: {}", df.shape)
