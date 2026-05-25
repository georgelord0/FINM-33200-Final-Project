import numpy as np
import pandas as pd

from bench.protocols import Panel
from bench.timesfm_wrappers import TimesFM


class _FakeTimesFM:
    def forecast(self, *, horizon, inputs):
        return np.full((len(inputs), horizon), 0.01), None

    def forecast_with_covariates(
        self,
        *,
        inputs,
        dynamic_numerical_covariates,
        dynamic_categorical_covariates,
        static_categorical_covariates,
        xreg_mode,
    ):
        self.last_covariates = dynamic_numerical_covariates
        return np.full((len(inputs), 1), 0.02), None


def _panel() -> Panel:
    dates = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.DataFrame(
        {
            "A": [0.1, 0.2, 0.3, 0.4, 0.5],
            "B": [0.1, np.nan, 0.3, 0.4, 0.5],
        },
        index=dates,
    )
    cov_rows = []
    for d in dates:
        for a in ["A", "B"]:
            cov_rows.append({"date": d, "asset_id": a, "x": 1.0})
    cov = pd.DataFrame(cov_rows).set_index(["date", "asset_id"])
    return Panel(returns=returns, covariates=cov, freq="B")


def test_timesfm_target_only_aligns_assets_without_imputation():
    model = TimesFM(context_len=4)
    model._model = _FakeTimesFM()
    out = model.predict(_panel(), pd.Timestamp("2024-01-05"), horizon=1)
    assert out.loc["A"] == 0.01
    assert pd.isna(out.loc["B"])


def test_timesfm_xreg_extends_covariates_one_step():
    model = TimesFM(context_len=3, use_xreg=True)
    fake = _FakeTimesFM()
    model._model = fake
    out = model.predict(_panel(), pd.Timestamp("2024-01-05"), horizon=1)
    assert out.loc["A"] == 0.02
    assert len(fake.last_covariates["x"][0]) == 4
