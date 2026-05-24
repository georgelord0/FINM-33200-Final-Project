"""
Forecast and portfolio metrics.

All functions take long-format frames with columns (date, asset_id, realized,
forecast) and return either a scalar or a small Series. They are pure
functions; no side effects, no global state.

The R^2 definition matches Gu et al. (2020) and Eq. 13 of Rahimikia/Ni/Wang
2025: it benchmarks against a zero prediction rather than the historical mean,
because the historical mean of individual stock excess returns is too noisy
to serve as a fair baseline.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TRADING_DAYS = 252


def r2_oos(df: pd.DataFrame) -> float:
    r = df["realized"].to_numpy()
    rhat = df["forecast"].to_numpy()
    ss_res = np.sum((r - rhat) ** 2)
    ss_tot = np.sum(r**2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mae(df: pd.DataFrame) -> float:
    return float(np.mean(np.abs(df["realized"] - df["forecast"])))


def rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean((df["realized"] - df["forecast"]) ** 2)))


def directional_accuracy(df: pd.DataFrame) -> float:
    realized_up = df["realized"] > 0
    forecast_up = df["forecast"] > 0
    return float((realized_up == forecast_up).mean())


def f1_direction(df: pd.DataFrame) -> float:
    y_true = (df["realized"] > 0).astype(int)
    y_pred = (df["forecast"] > 0).astype(int)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def long_short_returns(df: pd.DataFrame, n_deciles: int = 10) -> pd.Series:
    """Per-date L/S decile portfolio: long top decile by forecast, short bottom.

    Dates with too few assets or no forecast dispersion (e.g. the Zero
    baseline, where rank ties degenerate to alphabetical order) return NaN.
    """

    def ls(group: pd.DataFrame) -> float:
        if len(group) < n_deciles or group["forecast"].nunique() < 2:
            return np.nan
        ranks = group["forecast"].rank(method="first")
        top = ranks > (n_deciles - 1) * len(group) / n_deciles
        bot = ranks <= len(group) / n_deciles
        return group.loc[top, "realized"].mean() - group.loc[bot, "realized"].mean()

    return df.groupby("date", group_keys=False).apply(ls, include_groups=False).dropna()


def portfolio_stats(ls: pd.Series, bars_per_year: int = TRADING_DAYS) -> dict[str, float]:
    if len(ls) == 0:
        return {k: float("nan") for k in
                ["ann_return", "ann_vol", "sharpe", "max_dd", "max_dd_1d"]}
    mu = ls.mean()
    sd = ls.std(ddof=1)
    equity = (1 + ls).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "ann_return": mu * bars_per_year,
        "ann_vol": sd * np.sqrt(bars_per_year),
        "sharpe": (mu / sd) * np.sqrt(bars_per_year) if sd > 0 else float("nan"),
        "max_dd": float(drawdown.min()),
        "max_dd_1d": float(ls.min()),
    }


def summarize(df: pd.DataFrame, bars_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """Compute every reported metric on a long forecast frame.

    bars_per_year sets annualization for portfolio stats: 252 (daily, default),
    98280 = 390*252 (1-minute bars), 5896800 = 23400*252 (1-second bars).
    """
    ls = long_short_returns(df)
    out = {
        "r2_oos": r2_oos(df),
        "mae": mae(df),
        "rmse": rmse(df),
        "dir_acc": directional_accuracy(df),
        "f1": f1_direction(df),
        "n_obs": int(len(df)),
        "n_dates": int(df["date"].nunique()),
    }
    out.update(portfolio_stats(ls, bars_per_year=bars_per_year))
    return out
