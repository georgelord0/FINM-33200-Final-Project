"""
Expanding-window walk-forward driver.

For each OOS step, fit every model on data up to train_end and generate
h-step-ahead forecasts. Returns a long DataFrame with columns:
    date         realized-return date
    asset_id     stock
    model        forecaster name
    forecast     prediction made on the prior business day / bar
    realized     actual return on `date`

Two entry points:
    run(panel, models, oos_year, horizon) — calendar-year OOS slice; matches
        the daily replication setup.
    run_window(panel, models, train_end, oos_start, oos_end, horizon) —
        explicit timestamps. Use this for intraday (1s/1m) where a "year" is
        not the natural OOS unit.
"""

import pandas as pd

from .protocols import Forecaster, Panel


def run_window(
    panel: Panel,
    models: list[Forecaster],
    train_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    horizon: int = 1,
) -> pd.DataFrame:
    all_dates = panel.returns.index
    oos_dates = all_dates[(all_dates >= oos_start) & (all_dates <= oos_end)]

    rows = []
    for model in models:
        model.fit(panel, train_end)
        for fd in oos_dates:
            i = all_dates.get_loc(fd)
            if i < horizon:
                continue
            asof = all_dates[i - horizon]
            fcst = model.predict(panel, asof, horizon=horizon)
            realized = panel.returns.loc[fd]
            df = pd.DataFrame({"forecast": fcst, "realized": realized}).dropna()
            df["date"] = fd
            df["asset_id"] = df.index
            df["model"] = model.name
            rows.append(df.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True) if rows else _empty_frame()


def run(
    panel: Panel,
    models: list[Forecaster],
    oos_year: int,
    horizon: int = 1,
) -> pd.DataFrame:
    return run_window(
        panel,
        models,
        train_end=pd.Timestamp(f"{oos_year - 1}-12-31"),
        oos_start=pd.Timestamp(f"{oos_year}-01-01"),
        oos_end=pd.Timestamp(f"{oos_year}-12-31 23:59:59"),
        horizon=horizon,
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "asset_id", "model", "forecast", "realized"])
