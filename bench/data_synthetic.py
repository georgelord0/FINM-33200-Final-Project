"""
Synthetic intraday panel generator.

Useful for the latency benchmark (deterministic, no network I/O, scales to
any panel size) and for integration tests. Same AR(1) + common-factor
structure as tests/fixtures.synthetic_panel, generalised to arbitrary
bar frequency and length.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .covariates_intraday import build_intraday_covariates
from .protocols import Panel


def load_panel(
    start: str,
    end: str,
    n_assets: int,
    freq: str = "1min",
    phi: float = 0.05,
    common_factor_load: float = 0.3,
    idio_scale: float = 0.0005,
    seed: int = 0,
    covariate_kwargs: dict | None = None,
    cache_dir: str | Path | None = None,  # unused, accepted for dispatcher symmetry
) -> Panel:
    rng = np.random.default_rng(seed)
    # bdate_range only covers business days; intraday wants explicit ranges.
    dates = pd.date_range(start=start, end=end, freq=freq)
    if len(dates) < 2:
        raise ValueError(f"need at least 2 bars; got {len(dates)} for {start}..{end} @ {freq}")
    assets = [f"A{i:04d}" for i in range(n_assets)]

    common = rng.normal(scale=idio_scale * 0.5, size=len(dates))
    idio = rng.normal(scale=idio_scale, size=(len(dates), n_assets))
    r = np.zeros_like(idio)
    r[0] = idio[0]
    for t in range(1, len(dates)):
        r[t] = phi * r[t - 1] + common_factor_load * common[t] + idio[t]

    returns = pd.DataFrame(r, index=dates, columns=assets)
    volume = pd.DataFrame(
        rng.lognormal(mean=10, sigma=0.5, size=returns.shape).astype(np.int64),
        index=returns.index, columns=returns.columns,
    )
    covariates = build_intraday_covariates(returns, volume, **(covariate_kwargs or {}))
    return Panel(returns=returns, covariates=covariates, freq=freq)
