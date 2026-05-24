"""
Latency module: TimedForecaster wraps a Forecaster, captures fit and predict
timings without changing semantics, and summarize() produces the expected
percentile table.
"""

import pandas as pd
import pytest

from bench import baselines, latency, walkforward
from tests.fixtures import synthetic_panel


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(n_assets=15, n_years=3, seed=4)


def test_timed_forecaster_records_timings(panel):
    inner = baselines.Ridge()
    timed = latency.TimedForecaster(inner)
    train_end = pd.Timestamp("2020-12-31")
    timed.fit(panel, train_end)
    assert timed.record.fit_seconds >= 0.0

    asof = panel.returns.loc[:train_end].index[-1]
    timed.predict(panel, asof, horizon=1)
    assert len(timed.record.predict_seconds) == 1
    assert timed.record.predict_seconds[0] >= 0.0
    assert timed.record.n_assets_per_call[0] == 15


def test_timed_forecaster_does_not_change_forecasts(panel):
    train_end = pd.Timestamp("2020-12-31")
    bare = baselines.Ridge()
    timed = latency.TimedForecaster(baselines.Ridge())
    bare.fit(panel, train_end)
    timed.fit(panel, train_end)
    asof = panel.returns.loc[:train_end].index[-1]
    pd.testing.assert_series_equal(
        bare.predict(panel, asof), timed.predict(panel, asof),
    )


def test_summarize_percentiles_are_ordered(panel):
    models = [latency.TimedForecaster(baselines.Zero()),
              latency.TimedForecaster(baselines.Ridge())]
    walkforward.run(panel, models, oos_year=2020)
    df = latency.summarize([m.record for m in models], bars_per_year=252)
    assert set(df.columns) >= {"fit_seconds", "predict_mean", "p50", "p95", "p99",
                               "assets_per_sec", "bars_per_sec", "total_predict_seconds"}
    for _, row in df.iterrows():
        assert row["p50"] <= row["p95"] <= row["p99"]
        assert row["predict_mean"] >= 0
