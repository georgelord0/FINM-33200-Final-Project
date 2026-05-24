"""
Shared types for the benchmark harness.

Two things live here: the Panel dataclass that Max's data pipeline produces
(and that the harness consumes), and the Forecaster Protocol that George's
model wrappers implement. Both are intentionally narrow so plugging new
models or data sources in stays a one-file change.

PROVISIONAL: confirm with George before he commits to wrappers. Likely tweaks
are returning quantiles instead of a point forecast, or batching the predict
call across assets for GPU efficiency.
"""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Panel:
    returns: pd.DataFrame
    covariates: pd.DataFrame | None
    freq: str

    def assets(self) -> pd.Index:
        return self.returns.columns

    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index


class Forecaster(Protocol):
    name: str

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None: ...

    def predict(
        self, panel: Panel, asof: pd.Timestamp, horizon: int
    ) -> pd.Series: ...
