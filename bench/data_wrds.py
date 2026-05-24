"""
WRDS TAQ loader for 1-second / 1-minute intraday bars.

Reads WRDS_USERNAME / WRDS_PASSWORD from the env (loaded by the dispatcher
via python-dotenv). Pulls trades from the millisecond TAQ schema, resamples
server-side to the requested bar interval via `date_trunc`, and returns a
Panel matching the rest of the harness.

Schema layout (current WRDS millisecond TAQ):
    library:  taqm_<YYYY>
    trades:   ctm_YYYYMMDD     (columns: time_m, sym_root, price, size, ex, ...)

Server-side aggregation keeps the result small: even 1-second bars for one
session × 10 symbols is ~234K rows.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .covariates_intraday import build_intraday_covariates
from .protocols import Panel

_TRUNC = {"1s": "second", "1m": "minute"}


def _connect():
    try:
        import wrds
    except ImportError as e:
        raise ImportError(
            "wrds package not installed; `pip install wrds` or use a different source."
        ) from e
    user = os.environ.get("WRDS_USERNAME")
    pw = os.environ.get("WRDS_PASSWORD")
    if not user or not pw:
        raise RuntimeError(
            "WRDS_USERNAME / WRDS_PASSWORD not set. Add them to .env "
            "(see .env.example) and load via python-dotenv before running."
        )
    return wrds.Connection(wrds_username=user, wrds_password=pw)


def _fetch_session(db, session_date: pd.Timestamp, symbols: list[str], unit: str) -> pd.DataFrame:
    """Pull aggregated bars for one trading date. Resamples server-side."""
    yyyy = session_date.year
    yyyymmdd = session_date.strftime("%Y%m%d")
    sql_date = session_date.strftime("%Y-%m-%d")
    sql = f"""
        WITH base AS (
            SELECT (DATE '{sql_date}' + time_m)::timestamp AS ts,
                   sym_root, price, size
            FROM taqm_{yyyy}.ctm_{yyyymmdd}
            WHERE sym_root = ANY(%(syms)s)
              AND price > 0
              AND size > 0
        )
        SELECT date_trunc('{unit}', ts) AS bar_ts,
               sym_root,
               (array_agg(price ORDER BY ts DESC))[1] AS close,
               SUM(size) AS volume
        FROM base
        GROUP BY 1, 2
        ORDER BY 1, 2
    """
    return db.raw_sql(sql, params={"syms": symbols})


def _pivot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    close = df.pivot(index="bar_ts", columns="sym_root", values="close").sort_index()
    vol = df.pivot(index="bar_ts", columns="sym_root", values="volume").sort_index().fillna(0)
    return close, vol


def load_panel(
    start: str,
    end: str,
    cache_dir: str | Path,
    tickers: list[str],
    interval: str = "1s",
    covariate_kwargs: dict | None = None,
) -> Panel:
    if interval not in _TRUNC:
        raise ValueError(f"interval {interval!r} not supported; choose one of {sorted(_TRUNC)}")
    unit = _TRUNC[interval]
    cache_dir = Path(cache_dir) / "wrds"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = f"n{len(tickers)}_{interval}"
    cache = cache_dir / f"bars_{tag}_{start}_{end}.parquet"

    if cache.exists():
        bars = pd.read_parquet(cache)
    else:
        sessions = pd.bdate_range(start=start, end=end)
        if len(sessions) == 0:
            raise ValueError(f"no business days in [{start}, {end}]")
        db = _connect()
        try:
            frames = [_fetch_session(db, d, tickers, unit) for d in sessions]
        finally:
            try:
                db.close()
            except Exception:
                pass
        bars = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
        if bars.empty:
            raise RuntimeError(f"TAQ returned 0 bars for {tickers} {start}..{end}")
        bars.to_parquet(cache)

    close, vol = _pivot(bars)
    if close.empty:
        raise RuntimeError("no bars after pivot; check the cached parquet")

    # WRDS Postgres returns numeric as Decimal → pandas object dtype, which
    # breaks downstream torch / numpy code. Cast to float64 first.
    close = close.astype("float64")
    vol = vol.astype("float64")

    # Forward-fill: at 1s/1m resolution most bars have no trade. Standard
    # convention is "last-known price" — a missing bar means flat return.
    close = close.ffill()
    returns = np.log(close).diff()
    returns = returns.clip(-1.0, 1.0)
    returns = returns.iloc[1:]
    covariates = build_intraday_covariates(returns, vol.loc[returns.index], **(covariate_kwargs or {}))
    return Panel(returns=returns, covariates=covariates, freq=interval)
