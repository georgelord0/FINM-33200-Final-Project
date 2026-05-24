"""End-to-end smoke test on a synthetic panel."""

import pandas as pd
import pytest

from bench import baselines, metrics, walkforward
from tests.fixtures import synthetic_panel


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(n_assets=30, n_years=4, seed=0)


def test_walkforward_runs(panel):
    models = [baselines.Zero(), baselines.Mean(), baselines.Ridge(),
              baselines.LightGBM(n_estimators=20)]
    out = walkforward.run(panel, models, oos_year=2021)
    assert set(out.columns) == {"date", "asset_id", "model", "forecast", "realized"}
    assert set(out["model"].unique()) == {"zero", "mean", "ridge", "lightgbm"}
    assert (out.groupby("model")["date"].nunique() > 200).all()


def test_zero_baseline_has_r2_zero(panel):
    out = walkforward.run(panel, [baselines.Zero()], oos_year=2021)
    r2 = metrics.r2_oos(out)
    assert r2 == pytest.approx(0.0, abs=1e-12)


def test_lightgbm_r2_is_finite(panel):
    out = walkforward.run(panel, [baselines.LightGBM(n_estimators=20)], oos_year=2021)
    r2 = metrics.r2_oos(out)
    assert pd.notna(r2)
