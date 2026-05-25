import numpy as np
import pandas as pd

from bench.chronos_bolt import ChronosBolt
from bench.chronos_two import ChronosTwo
from bench.moirai import Moirai
from bench.protocols import Panel


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return _FakeTensor(self.value[key])

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def item(self):
        return float(self.value)


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def tensor(value, dtype=None):
        return _FakeTensor(value)


class _FakeBoltPipeline:
    def predict_quantiles(self, ctx, prediction_length, quantile_levels):
        n = ctx.value.shape[0]
        return None, _FakeTensor(np.full((n, prediction_length), 0.03))


class _FakeChronosTwoPipeline:
    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        means = [_FakeTensor(np.full((1, prediction_length), 0.04)) for _ in inputs]
        return None, means


class _FakeForecast:
    mean = np.array([0.05])


class _FakePredictor:
    def predict(self, ds):
        return [_FakeForecast() for _ in ds]


def _panel() -> Panel:
    dates = pd.bdate_range("2024-01-01", periods=4)
    returns = pd.DataFrame(
        {"A": [0.1, 0.2, 0.3, 0.4], "B": [0.1, np.nan, 0.3, 0.4]},
        index=dates,
    )
    rows = []
    for d in dates:
        for a in ["A", "B"]:
            rows.append({"date": d, "asset_id": a, "x": 1.0})
    cov = pd.DataFrame(rows).set_index(["date", "asset_id"])
    return Panel(returns=returns, covariates=cov, freq="B")


def test_chronos_bolt_aligns_assets_without_zero_fill():
    model = ChronosBolt(context_len=4)
    model._pipeline = _FakeBoltPipeline()
    model._torch = _FakeTorch()
    out = model.predict(_panel(), pd.Timestamp("2024-01-04"), horizon=1)
    assert out.loc["A"] == 0.03
    assert pd.isna(out.loc["B"])


def test_chronos_two_uses_covariates_and_aligns_assets():
    model = ChronosTwo(context_len=4)
    model._pipeline = _FakeChronosTwoPipeline()
    model._torch = _FakeTorch()
    out = model.predict(_panel(), pd.Timestamp("2024-01-04"), horizon=1)
    assert out.loc["A"] == 0.04
    assert pd.isna(out.loc["B"])


def test_moirai_aligns_assets_without_loading_weights():
    model = Moirai(context_len=4)
    model._predictor = _FakePredictor()
    model._prediction_length = 1
    model._pandas_dataset = lambda data: list(data)
    out = model.predict(_panel(), pd.Timestamp("2024-01-04"), horizon=1)
    assert out.loc["A"] == 0.05
    assert pd.isna(out.loc["B"])
