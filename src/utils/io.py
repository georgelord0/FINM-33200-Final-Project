"""I/O utilities for the src pipeline.

Provides thin wrappers around YAML config loading and Parquet read/write so
that every module in the project uses a consistent, well-logged interface.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

# Walk up from this file to find the project root (contains pyproject.toml or
# the top-level src package).
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent  # utils -> src -> project root


def _find_configs_dir() -> Path:
    """Locate the configs/ directory.

    Checks (in order):
    1. ``src/configs/`` inside the project root.
    2. ``configs/`` at the project root.
    """
    for candidate in (
        _PROJECT_ROOT / "src" / "configs",
        _PROJECT_ROOT / "configs",
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find a configs/ directory under {_PROJECT_ROOT}"
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(name: str) -> dict[str, Any]:
    """Load a YAML configuration file by *name*.

    The ``.yaml`` extension is appended automatically if not present.

    Parameters
    ----------
    name:
        Base name of the config file (e.g. ``"assets"`` or ``"paths.yaml"``).

    Returns
    -------
    dict[str, Any]
        Parsed YAML contents.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"

    configs_dir = _find_configs_dir()
    path = configs_dir / name

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)

    log.debug("Loaded config '{}' from {}", name, path)
    return data


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    partition_cols: list[str] | None = None,
) -> None:
    """Save a DataFrame as a Parquet file (or partitioned dataset).

    Parent directories are created automatically.

    Parameters
    ----------
    df:
        The DataFrame to persist.
    path:
        Destination path.  When *partition_cols* is given this should be a
        directory; otherwise a single ``.parquet`` file is written.
    partition_cols:
        Optional column names to partition on.  Each unique combination of
        values creates a sub-directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)

    if partition_cols:
        path.mkdir(parents=True, exist_ok=True)
        pq.write_to_dataset(
            table,
            root_path=str(path),
            partition_cols=partition_cols,
        )
        log.info(
            "Saved partitioned parquet ({} rows) to {} [partitions={}]",
            len(df),
            path,
            partition_cols,
        )
    else:
        pq.write_table(table, str(path))
        log.info("Saved parquet ({} rows, {:.1f} MB) to {}", len(df), path.stat().st_size / 1e6, path)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file or partitioned Parquet directory.

    Parameters
    ----------
    path:
        Path to a ``.parquet`` file **or** a directory containing a
        partitioned dataset.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")

    if path.is_dir():
        dataset = pq.ParquetDataset(str(path))
        df: pd.DataFrame = dataset.read().to_pandas()
    else:
        df = pq.read_table(str(path)).to_pandas()

    log.debug("Loaded parquet ({} rows, {} cols) from {}", len(df), len(df.columns), path)
    return df


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_data_dir() -> Path:
    """Return the project data directory.

    Resolution order:
    1. ``TSFM_DATA_DIR`` environment variable.
    2. ``data_dir`` key in ``configs/paths.yaml``.
    3. Falls back to ``<project_root>/src/data``.

    Returns
    -------
    Path
        Absolute path to the data directory (created if it does not exist).
    """
    env_dir = os.environ.get("TSFM_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    else:
        try:
            paths_cfg = load_config("paths")
            rel = paths_cfg.get("data_dir", "data")
            data_dir = _PROJECT_ROOT / "src" / rel
        except FileNotFoundError:
            data_dir = _PROJECT_ROOT / "src" / "data"

    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    log.debug("Data directory: {}", data_dir)
    return data_dir


def ensure_dir(path: Path) -> Path:
    """Create *path* and all parents if they do not exist (``mkdir -p``).

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The same *path*, guaranteed to exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
