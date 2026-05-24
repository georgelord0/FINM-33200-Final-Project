"""
Fama-French 25 size–BM portfolios, daily.

Pulls Ken French's daily 5x5 portfolio file, parses the value-weighted block,
caches the long-format frame, and returns a Panel compatible with the rest of
the harness. This is one of the only daily-frequency datasets covered by the
FTSFR benchmark (Bejarano et al. 2025), so results here are directly
comparable to their published baselines.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .protocols import Panel

FF25_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "25_Portfolios_5x5_Daily_CSV.zip"
)


def _download_csv(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "ff25_daily.csv"
    if csv_path.exists():
        return csv_path
    req = urllib.request.Request(FF25_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        zbytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        name = zf.namelist()[0]
        csv_path.write_bytes(zf.read(name))
    return csv_path


def _parse_ff25(csv_path: Path) -> pd.DataFrame:
    """Ken French's file packs four panels (VW with/without divs, EW
    with/without divs) separated by blank lines, with a leading licence
    header. We keep the first panel: Average Value Weighted Returns - Daily.
    """
    with csv_path.open() as f:
        lines = f.readlines()
    # Find the header line: first line whose first token is an 8-digit date.
    start = next(i for i, line in enumerate(lines)
                 if line.split(",")[0].strip().isdigit()
                 and len(line.split(",")[0].strip()) == 8)
    # The header is one line above.
    header = lines[start - 1].strip().split(",")
    columns = ["date"] + [c.strip() for c in header if c.strip()]

    end = start
    while end < len(lines) and lines[end].strip():
        if not lines[end].split(",")[0].strip().isdigit():
            break
        end += 1
    block = "".join(lines[start:end])

    df = pd.read_csv(io.StringIO(block), header=None, names=columns)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df.set_index("date").sort_index() / 100.0


def _build_covariates(returns: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for lag in (1, 5, 21):
        pieces.append(returns.shift(lag).stack().rename(f"ret_lag{lag}"))
    pieces.append(returns.rolling(21).std().shift(1).stack().rename("vol_21"))
    mom = returns.rolling(252).sum().shift(21) - returns.rolling(21).sum().shift(1)
    pieces.append(mom.stack().rename("mom_12_1"))
    cov = pd.concat(pieces, axis=1)
    cov.index.names = ["date", "asset_id"]
    return cov


def load_panel(start: str, end: str, cache_dir: str | Path) -> Panel:
    cache_dir = Path(cache_dir)
    csv = _download_csv(cache_dir)
    returns = _parse_ff25(csv)
    returns = returns.loc[start:end].clip(-1.0, 1.0)
    covariates = _build_covariates(returns)
    return Panel(returns=returns, covariates=covariates, freq="B")