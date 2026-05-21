"""
Synthetic panel for tests. NOT for production use; the CLI must read real
data and error loud if it's missing (rubric rule).

Builds an AR(1) cross-section with mild common factor structure so smoke tests
exercise the harness end-to-end without needing CRSP or yfinance.
"""

import numpy as np
import pandas as pd

from bench.protocols import Panel


def synthetic_panel(
    n_assets: int = 50,
    n_years: int = 5,
    phi: float = 0.05,
    common_factor_load: float = 0.3,
    seed: int = 0,
) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_years * 252)
    assets = [f"A{i:03d}" for i in range(n_assets)]

    common = rng.normal(scale=0.01, size=len(dates))
    idio = rng.normal(scale=0.015, size=(len(dates), n_assets))
    r = np.zeros_like(idio)
    r[0] = idio[0]
    for t in range(1, len(dates)):
        r[t] = phi * r[t - 1] + common_factor_load * common[t] + idio[t]

    returns = pd.DataFrame(r, index=dates, columns=assets)

    cov = []
    for lag in (1, 5, 21):
        cov.append(returns.shift(lag).stack().rename(f"ret_lag{lag}"))
    cov.append(returns.rolling(21).std().shift(1).stack().rename("vol_21"))
    cov_df = pd.concat(cov, axis=1)
    cov_df.index.names = ["date", "asset_id"]

    return Panel(returns=returns, covariates=cov_df, freq="B")
