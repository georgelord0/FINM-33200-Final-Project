"""
Backward-only covariates for intraday bar panels.

Direct analogue of `bench.data_yfinance._build_covariates`, but with windows
measured in *bars* rather than business days. Same shape (ret lags, realized
vol, momentum spread, log-volume z-score), so the leakage guards in
tests/test_leakage.py keep working without modification.

Defaults are tuned for 1-minute bars (390 bars/session):
    lags=(1, 5, 60)        # 1m, 5m, 1h
    vol_window=60          # 1h realized vol
    mom_long=390           # one trading day momentum
    mom_short=60           # less one hour
    log_vol_z_window=390   # one-day rolling z of log volume

For 1-second bars (23400 bars/session) pass bar-count equivalents, e.g.
    lags=(1, 60, 600), vol_window=60, mom_long=23400, mom_short=60,
    log_vol_z_window=23400.
"""

import numpy as np
import pandas as pd


def build_intraday_covariates(
    returns: pd.DataFrame,
    volume: pd.DataFrame | None,
    *,
    lags: tuple[int, ...] = (1, 5, 60),
    vol_window: int = 60,
    mom_long: int = 390,
    mom_short: int = 60,
    log_vol_z_window: int = 390,
) -> pd.DataFrame:
    pieces = []
    for lag in lags:
        pieces.append(returns.shift(lag).stack().rename(f"ret_lag{lag}"))
    pieces.append(
        returns.rolling(vol_window).std().shift(1).stack().rename(f"vol_{vol_window}")
    )
    mom = returns.rolling(mom_long).sum().shift(mom_short) - returns.rolling(mom_short).sum().shift(1)
    pieces.append(mom.stack().rename(f"mom_{mom_long}_{mom_short}"))
    if volume is not None:
        logv = np.log(volume.replace(0, np.nan))
        z = (logv - logv.rolling(log_vol_z_window).mean()) / logv.rolling(log_vol_z_window).std()
        pieces.append(z.shift(1).stack().rename("log_vol_z"))

    cov = pd.concat(pieces, axis=1)
    cov.index.names = ["date", "asset_id"]
    return cov
