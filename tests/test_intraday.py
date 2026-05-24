"""
Intraday harness checks: synthetic 1-min panel runs end-to-end through
walkforward.run_window, intraday covariates are leakage-free, and the
bars_per_year knob actually scales portfolio annualisation as expected.
"""

import numpy as np
import pandas as pd
import pytest

from bench import baselines, data_synthetic, metrics, walkforward
from bench.covariates_intraday import build_intraday_covariates


@pytest.fixture(scope="module")
def intraday_panel():
    # ~4 sessions × 390 1-min bars ≈ 1560 bars; enough for rolling windows.
    return data_synthetic.load_panel(
        start="2025-01-02 09:30", end="2025-01-08 16:00",
        n_assets=20, freq="1min", seed=7,
    )


def test_walkforward_window_runs(intraday_panel):
    train_end = pd.Timestamp("2025-01-07 16:00")
    oos_start = pd.Timestamp("2025-01-08 09:30")
    oos_end = pd.Timestamp("2025-01-08 16:00")
    models = [baselines.Zero(), baselines.Ridge(), baselines.LightGBM(n_estimators=20)]
    out = walkforward.run_window(
        intraday_panel, models, train_end=train_end,
        oos_start=oos_start, oos_end=oos_end, horizon=1,
    )
    assert set(out.columns) == {"date", "asset_id", "model", "forecast", "realized"}
    assert set(out["model"].unique()) == {"zero", "ridge", "lightgbm"}
    assert out["forecast"].notna().all()
    assert out["date"].min() >= oos_start
    assert out["date"].max() <= oos_end


def test_intraday_covariates_only_use_past():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-02 09:30", periods=500, freq="1min")
    cols = ["A", "B", "C"]
    r = pd.DataFrame(rng.normal(scale=0.0005, size=(500, 3)), index=idx, columns=cols)
    v = pd.DataFrame(rng.lognormal(10, 0.5, size=(500, 3)).astype(int), index=idx, columns=cols)

    cov_clean = build_intraday_covariates(r, v)
    cutoff = r.index[len(r) // 2]
    r_corr = r.copy(); r_corr.loc[r_corr.index > cutoff] = 99.0
    v_corr = v.copy(); v_corr.loc[v_corr.index > cutoff] = 99.0
    cov_corr = build_intraday_covariates(r_corr, v_corr)

    before = cov_clean.loc[cov_clean.index.get_level_values("date") <= cutoff]
    after = cov_corr.loc[cov_corr.index.get_level_values("date") <= cutoff]
    pd.testing.assert_frame_equal(before, after)


def test_bars_per_year_scales_annualisation():
    # Same L/S returns; annualised vol must scale by sqrt(ratio).
    rng = np.random.default_rng(1)
    ls = pd.Series(rng.normal(0.0, 1e-4, size=500))
    daily = metrics.portfolio_stats(ls, bars_per_year=252)
    intraday = metrics.portfolio_stats(ls, bars_per_year=98280)
    ratio = (98280 / 252) ** 0.5
    assert intraday["ann_vol"] == pytest.approx(daily["ann_vol"] * ratio, rel=1e-9)
    assert intraday["sharpe"] == pytest.approx(daily["sharpe"] * ratio, rel=1e-9)
