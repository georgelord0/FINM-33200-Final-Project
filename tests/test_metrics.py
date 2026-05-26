"""Hand-computed checks for each metric."""

import numpy as np
import pandas as pd
import pytest

from bench import metrics as m


@pytest.fixture
def forecasts():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03"] * 4 + ["2022-01-04"] * 4),
        "asset_id": ["A", "B", "C", "D"] * 2,
        "realized": [0.01, -0.02, 0.03, -0.01, 0.02, -0.01, -0.03, 0.04],
        "forecast": [0.02, -0.01, 0.04, 0.01, 0.01, -0.02, -0.02, 0.05],
    })


def test_portfolio_stats_survives_full_wipeout_bar():
    # bar <= -1 used to make cumprod go negative and max_dd nonsense
    ls = pd.Series([0.01, -1.5, 0.02, -0.005, 0.01])
    s = m.portfolio_stats(ls)
    assert s["max_dd"] >= -1.0
    assert s["max_dd_1d"] == pytest.approx(-1.5)
    assert np.isfinite(s["sharpe"])


def test_r2_oos_zero_forecast_is_zero():
    df = pd.DataFrame({
        "realized": np.random.default_rng(0).normal(size=100),
        "forecast": np.zeros(100),
    })
    assert m.r2_oos(df) == pytest.approx(0.0)


def test_r2_oos_perfect_forecast_is_one():
    r = np.array([0.01, -0.02, 0.03, -0.01])
    df = pd.DataFrame({"realized": r, "forecast": r})
    assert m.r2_oos(df) == pytest.approx(1.0)


def test_directional_accuracy(forecasts):
    # 7 of 8 signs match (only D on day 1 disagrees)
    assert m.directional_accuracy(forecasts) == pytest.approx(7 / 8)


def test_f1_runs(forecasts):
    f1 = m.f1_direction(forecasts)
    assert 0.0 <= f1 <= 1.0


def test_long_short_returns_two_asset():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03"] * 2),
        "asset_id": ["A", "B"],
        "realized": [0.05, -0.05],
        "forecast": [0.10, -0.10],
    })
    # With n_deciles=10 and only 2 assets per date, len < n_deciles → NaN
    ls = m.long_short_returns(df, n_deciles=10)
    assert ls.isna().all() or len(ls) == 0


def test_long_short_returns_n2_deciles():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03"] * 4),
        "asset_id": ["A", "B", "C", "D"],
        "realized": [0.05, 0.02, -0.02, -0.05],
        "forecast": [0.10, 0.05, -0.05, -0.10],
    })
    ls = m.long_short_returns(df, n_deciles=2)
    # top half (A,B avg=0.035) minus bottom half (C,D avg=-0.035) = 0.07
    assert ls.iloc[0] == pytest.approx(0.07)


def test_portfolio_stats_smoke():
    rng = np.random.default_rng(0)
    ls = pd.Series(rng.normal(0.0005, 0.01, size=252))
    s = m.portfolio_stats(ls)
    assert set(s) == {"ann_return", "ann_vol", "sharpe", "max_dd", "max_dd_1d"}
    assert all(np.isfinite(v) for v in s.values())


def test_summarize_smoke(forecasts):
    s = m.summarize(forecasts)
    assert s["n_obs"] == 8
    assert s["n_dates"] == 2
    assert "sharpe" in s
