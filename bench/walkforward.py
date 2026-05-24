"""
Expanding-window walk-forward driver.

For each OOS year, fit every model on data up to the prior year-end and
generate one-step-ahead forecasts for every business day in the OOS year.
Returns a long DataFrame with columns:
    date         realized-return date
    asset_id     stock
    model        forecaster name
    forecast     prediction made on the prior business day
    realized     actual return on `date`
"""

import pandas as pd

from .protocols import Forecaster, Panel


def run(
    panel: Panel,
    models: list[Forecaster],
    oos_year: int,
    horizon: int = 1,
) -> pd.DataFrame:
    train_end = pd.Timestamp(f"{oos_year - 1}-12-31")
    oos_dates = panel.returns.loc[f"{oos_year}-01-01":f"{oos_year}-12-31"].index
    all_dates = panel.returns.index

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


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "asset_id", "model", "forecast", "realized"])
