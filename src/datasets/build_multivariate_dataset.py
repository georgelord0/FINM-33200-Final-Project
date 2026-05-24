"""Build multivariate tensor datasets for Time-Series Foundation Models.

Converts the canonical panel dataset (produced by
:mod:`src.datasets.build_panel_dataset`) into fixed-length
context-window tensors suitable for TSFM training and evaluation.

Each sample consists of ``context_window`` days of multi-feature history for
a single asset, paired with the next-period excess return as the target.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.constants import (
    COL_DATE,
    COL_PERMNO,
)
from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default feature groups
# ---------------------------------------------------------------------------

_DEFAULT_FEATURE_COLS: list[str] = [
    # Returns
    "ret",
    "excess_ret",
    # Momentum
    "mom_21",
    "mom_63",
    "mom_126",
    "mom_252",
    # Volatility
    "vol_21",
    "vol_63",
    # Liquidity
    "illiq_21",
    "illiq_63",
    # Size
    "log_market_cap",
    # Fundamentals
    "book_to_market",
    "roe",
    "asset_growth",
    "gross_profitability",
    # Macro / factor
    "mktrf",
    "smb",
    "hml",
]

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_multivariate(
    panel_path: str,
    context_window: int = 252,
    horizon: int = 1,
    feature_cols: list[str] | None = None,
    nan_threshold: float = 0.20,
) -> dict:
    """Build multivariate tensor dataset from the panel.

    For every valid (permno, date) prediction point, extracts a
    ``context_window``-length history of features and a forward-looking
    excess-return target.

    Parameters
    ----------
    panel_path:
        Path to the panel parquet dataset.  Can be a single ``.parquet``
        file or a directory of partitioned parquet files.
    context_window:
        Number of historical trading days used as model context.
    horizon:
        Prediction horizon in trading days.  The target is the excess
        return ``horizon`` steps ahead.  Default is 1 (next-day).
    feature_cols:
        Ordered list of feature columns to include.  If ``None``, a
        default set of return, momentum, volatility, liquidity,
        fundamental, and macro features is used.
    nan_threshold:
        Maximum fraction of NaN values allowed in a sample's context
        window.  Samples exceeding this threshold are dropped.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"X"``: ``np.ndarray`` of shape ``(n_samples, context_window, n_features)``
        - ``"y"``: ``np.ndarray`` of shape ``(n_samples,)``
        - ``"dates"``: ``np.ndarray`` of prediction dates
        - ``"permnos"``: ``np.ndarray`` of asset identifiers
        - ``"feature_names"``: ``list[str]`` of feature column names
    """
    # ------------------------------------------------------------------
    # 1. Load panel
    # ------------------------------------------------------------------
    panel_path_obj = Path(panel_path)
    if panel_path_obj.is_dir():
        log.info("Loading partitioned panel from {}", panel_path_obj)
        panel = pd.read_parquet(panel_path_obj)
    else:
        log.info("Loading panel from {}", panel_path_obj)
        panel = pd.read_parquet(panel_path_obj)

    panel[COL_DATE] = pd.to_datetime(panel[COL_DATE])
    panel[COL_PERMNO] = panel[COL_PERMNO].astype(int)
    panel.sort_values([COL_PERMNO, COL_DATE], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    log.info("Panel loaded: {} rows, {} columns", len(panel), len(panel.columns))

    # ------------------------------------------------------------------
    # 2. Resolve feature columns
    # ------------------------------------------------------------------
    if feature_cols is None:
        feature_cols = [c for c in _DEFAULT_FEATURE_COLS if c in panel.columns]
        if not feature_cols:
            # Fall back to all numeric columns that are not IDs or target
            exclude = {COL_DATE, COL_PERMNO, "target_ret", "sector", "year"}
            feature_cols = [
                c
                for c in panel.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
    else:
        missing = [c for c in feature_cols if c not in panel.columns]
        if missing:
            log.warning("Requested feature columns not in panel: {}", missing)
            feature_cols = [c for c in feature_cols if c in panel.columns]

    n_features = len(feature_cols)
    log.info("Using {} features: {}", n_features, feature_cols)

    if n_features == 0:
        raise ValueError("No valid feature columns found in the panel")

    # ------------------------------------------------------------------
    # 3. Build samples per permno
    # ------------------------------------------------------------------
    max_nan_count = int(context_window * n_features * nan_threshold)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    date_list: list[pd.Timestamp] = []
    permno_list: list[int] = []

    permnos = panel[COL_PERMNO].unique()
    log.info("Building tensors for {} assets ...", len(permnos))

    skipped_short = 0
    skipped_nan = 0

    for permno in tqdm(permnos, desc="Building multivariate samples"):
        asset = panel[panel[COL_PERMNO] == permno].copy()

        if len(asset) < context_window + horizon:
            skipped_short += 1
            continue

        dates = asset[COL_DATE].values
        features = asset[feature_cols].values  # shape: (T, n_features)
        targets = asset["target_ret"].values

        # Slide over valid prediction points
        for t in range(context_window, len(asset) - horizon + 1):
            # Context: [t - context_window, t)
            ctx = features[t - context_window : t]  # (context_window, n_features)

            # Check NaN threshold
            nan_count = np.isnan(ctx).sum()
            if nan_count > max_nan_count:
                skipped_nan += 1
                continue

            # Target: excess return at t + horizon - 1
            target_val = targets[t + horizon - 1] if horizon > 1 else targets[t]
            if np.isnan(target_val):
                continue

            # Fill remaining NaNs with 0
            ctx_clean = np.nan_to_num(ctx, nan=0.0)

            X_list.append(ctx_clean)
            y_list.append(target_val)
            date_list.append(pd.Timestamp(dates[t]))
            permno_list.append(permno)

    log.info(
        "Skipped {} assets (too short), {} samples (NaN threshold exceeded)",
        skipped_short,
        skipped_nan,
    )

    # ------------------------------------------------------------------
    # 4. Stack into arrays
    # ------------------------------------------------------------------
    if not X_list:
        raise ValueError("No valid samples were constructed. Check your data and parameters.")

    X = np.stack(X_list, axis=0)  # (n_samples, context_window, n_features)
    y = np.array(y_list, dtype=np.float64)  # (n_samples,)
    dates_arr = np.array(date_list)
    permnos_arr = np.array(permno_list, dtype=np.int64)

    log.info("--- Tensor Summary ---")
    log.info("X shape: {}", X.shape)
    log.info("y shape: {}", y.shape)
    log.info(
        "Date range: {} to {}",
        pd.Timestamp(dates_arr.min()).strftime("%Y-%m-%d"),
        pd.Timestamp(dates_arr.max()).strftime("%Y-%m-%d"),
    )
    log.info("Unique permnos: {}", len(np.unique(permnos_arr)))
    log.info("y mean: {:.6f}, std: {:.6f}", y.mean(), y.std())

    data = {
        "X": X,
        "y": y,
        "dates": dates_arr,
        "permnos": permnos_arr,
        "feature_names": feature_cols,
    }

    return data


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------


def split_by_date(
    data: dict,
    train_end: str,
    val_end: str,
) -> tuple[dict, dict, dict]:
    """Split a multivariate dataset by date into train / val / test sets.

    Uses a strict temporal split with no shuffling to prevent look-ahead
    bias.

    Parameters
    ----------
    data:
        Dictionary returned by :func:`build_multivariate` with keys
        ``"X"``, ``"y"``, ``"dates"``, ``"permnos"``, ``"feature_names"``.
    train_end:
        Last date (inclusive) of the training set, as an ISO string
        (e.g. ``"2018-12-31"``).
    val_end:
        Last date (inclusive) of the validation set, as an ISO string
        (e.g. ``"2020-12-31"``).  Dates after this become the test set.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(train, val, test)`` dictionaries, each with the same keys as
        *data*.
    """
    dates = pd.to_datetime(data["dates"])
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train_mask = dates <= train_end_ts
    val_mask = (dates > train_end_ts) & (dates <= val_end_ts)
    test_mask = dates > val_end_ts

    def _subset(mask: np.ndarray) -> dict:
        return {
            "X": data["X"][mask],
            "y": data["y"][mask],
            "dates": data["dates"][mask],
            "permnos": data["permnos"][mask],
            "feature_names": data["feature_names"],
        }

    train = _subset(train_mask.values)
    val = _subset(val_mask.values)
    test = _subset(test_mask.values)

    log.info(
        "Split sizes: train={}, val={}, test={}",
        len(train["y"]),
        len(val["y"]),
        len(test["y"]),
    )

    return train, val, test


# ---------------------------------------------------------------------------
# Save / load utilities
# ---------------------------------------------------------------------------


def save_dataset(data: dict, path: str | Path) -> None:
    """Save a multivariate dataset as a compressed ``.npz`` file.

    Parameters
    ----------
    data:
        Dictionary returned by :func:`build_multivariate`.
    path:
        Output file path (should end with ``.npz``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        X=data["X"],
        y=data["y"],
        dates=data["dates"],
        permnos=data["permnos"],
        feature_names=np.array(data["feature_names"], dtype=object),
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    log.info("Saved dataset to {} ({:.1f} MB)", path, size_mb)


def load_dataset(path: str | Path) -> dict:
    """Load a multivariate dataset from a ``.npz`` file.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file saved by :func:`save_dataset`.

    Returns
    -------
    dict
        Dictionary with the same keys as returned by
        :func:`build_multivariate`.
    """
    npz = np.load(path, allow_pickle=True)
    data = {
        "X": npz["X"],
        "y": npz["y"],
        "dates": npz["dates"],
        "permnos": npz["permnos"],
        "feature_names": list(npz["feature_names"]),
    }
    log.info("Loaded dataset from {}: X={}", path, data["X"].shape)
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    from src.utils.logging_utils import setup_logger

    parser = argparse.ArgumentParser(
        description="Build multivariate tensor dataset for TSFMs"
    )
    parser.add_argument(
        "--panel-path",
        type=str,
        default=None,
        help="Path to panel.parquet (file or directory). "
        "Defaults to data/datasets/panel.parquet",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=252,
        help="Number of historical days in the context window (default: 252)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in trading days (default: 1)",
    )
    parser.add_argument(
        "--nan-threshold",
        type=float,
        default=0.20,
        help="Max fraction of NaN values allowed per sample (default: 0.20)",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Last training date (ISO format) for optional split",
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default=None,
        help="Last validation date (ISO format) for optional split",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .npz files. Defaults to data/datasets/",
    )
    args = parser.parse_args()

    setup_logger(level="INFO")

    # Resolve defaults
    root = _project_root()
    panel_path = args.panel_path or str(root / "data" / "datasets" / "panel.parquet")
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "datasets"

    # Build
    data = build_multivariate(
        panel_path=panel_path,
        context_window=args.context_window,
        horizon=args.horizon,
        nan_threshold=args.nan_threshold,
    )

    # Save full dataset
    out_name = f"multivariate_cw{args.context_window}_h{args.horizon}.npz"
    save_dataset(data, output_dir / out_name)

    # Optionally split and save
    if args.train_end and args.val_end:
        train, val, test = split_by_date(data, args.train_end, args.val_end)
        save_dataset(train, output_dir / f"train_{out_name}")
        save_dataset(val, output_dir / f"val_{out_name}")
        save_dataset(test, output_dir / f"test_{out_name}")

    log.info("Done.")
