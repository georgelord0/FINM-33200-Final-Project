"""Plots for the writeup: Sharpe-by-model bar, cumulative L/S equity curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import metrics


def sharpe_by_model(out_df: pd.DataFrame, path: Path,
                    bars_per_year: int = metrics.TRADING_DAYS) -> None:
    sharpes = {}
    for name, group in out_df.groupby("model", sort=False):
        ls = metrics.long_short_returns(group)
        sharpes[name] = metrics.portfolio_stats(ls, bars_per_year=bars_per_year)["sharpe"]
    s = pd.Series(sharpes).sort_values()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(s.index, s.values, color="0.3")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("Annualised Sharpe (L/S decile portfolio)")
    ax.set_title("Sharpe by model")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def equity_curves(out_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for name, group in out_df.groupby("model", sort=False):
        ls = metrics.long_short_returns(group)
        if ls.empty:
            continue
        equity = (1 + ls).cumprod()
        ax.plot(equity.index, equity.values, label=name, lw=1.2)
    ax.axhline(1, color="0.7", lw=0.8)
    ax.set_ylabel("Cumulative L/S return")
    ax.set_title("Long–short decile equity curves")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_all(out_df: pd.DataFrame, out_dir: Path,
              bars_per_year: int = metrics.TRADING_DAYS) -> None:
    sharpe_by_model(out_df, out_dir / "sharpe_by_model.png", bars_per_year=bars_per_year)
    equity_curves(out_df, out_dir / "equity_curves.png")
