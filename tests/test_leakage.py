"""
Leakage guards.

These tests assert two invariants that the walk-forward harness depends on:
(1) covariate row at date t is a function of data <= t-1 only (so swapping
    out future returns leaves earlier covariate rows unchanged);
(2) refitting a baseline after corrupting OOS-year returns yields identical
    coefficients (training set is strictly pre-OOS).

If either fails, every result we report is contaminated.
"""

import numpy as np
import pandas as pd
import pytest

from bench import baselines, walkforward
from bench.data_yfinance import _build_covariates
from tests.fixtures import synthetic_panel


def _synthetic_with_volume(seed: int = 0):
    panel = synthetic_panel(n_assets=20, n_years=4, seed=seed)
    # synthetic_panel doesn't fabricate volume; build a plausible one.
    rng = np.random.default_rng(seed + 1)
    volume = pd.DataFrame(
        rng.lognormal(mean=15, sigma=0.4, size=panel.returns.shape).astype(int),
        index=panel.returns.index, columns=panel.returns.columns,
    )
    return panel.returns, volume


def test_covariates_only_use_past():
    r, v = _synthetic_with_volume()
    cov_clean = _build_covariates(r, v)

    cutoff = r.index[len(r) // 2]
    r_corrupted = r.copy()
    r_corrupted.loc[r_corrupted.index > cutoff] = 99.0
    v_corrupted = v.copy()
    v_corrupted.loc[v_corrupted.index > cutoff] = 99.0
    cov_corrupted = _build_covariates(r_corrupted, v_corrupted)

    before = cov_clean.loc[cov_clean.index.get_level_values("date") <= cutoff]
    after = cov_corrupted.loc[cov_corrupted.index.get_level_values("date") <= cutoff]
    pd.testing.assert_frame_equal(before, after)


def test_ridge_fit_ignores_oos_returns():
    """Train two Ridges on identical-up-to-train_end panels that diverge
    *only* after train_end. Backward-only features + train-end cutoff means
    the fitted coefficients must match bit-for-bit."""
    from bench.protocols import Panel
    r, v = _synthetic_with_volume()
    train_end = pd.Timestamp("2020-12-31")

    cov_clean = _build_covariates(r, v)
    p_clean = Panel(returns=r, covariates=cov_clean, freq="B")

    r_corrupted = r.copy()
    r_corrupted.loc[r_corrupted.index > train_end] = 99.0
    v_corrupted = v.copy()
    v_corrupted.loc[v_corrupted.index > train_end] = 99.0
    cov_corrupted = _build_covariates(r_corrupted, v_corrupted)
    p_corrupted = Panel(returns=r_corrupted, covariates=cov_corrupted, freq="B")

    m_clean = baselines.Ridge()
    m_clean.fit(p_clean, train_end)
    m_corrupted = baselines.Ridge()
    m_corrupted.fit(p_corrupted, train_end)

    np.testing.assert_allclose(
        m_clean._model.coef_, m_corrupted._model.coef_,
        rtol=1e-10, atol=1e-12,
    )


def test_walkforward_predict_only_sees_past():
    """Predictions for OOS year do not change if we corrupt OOS-future returns
    *after* the date being predicted. (Subtle: the covariate frame is built
    once at load time, so this confirms the harness re-uses precomputed
    backward-only features rather than refreshing them with future data.)"""
    panel = synthetic_panel(n_assets=15, n_years=3, seed=2)
    out_clean = walkforward.run(panel, [baselines.Ridge()], oos_year=2020)

    pivot = pd.Timestamp("2020-06-30")
    panel_post = synthetic_panel(n_assets=15, n_years=3, seed=2)
    panel_post.returns.loc[panel_post.returns.index > pivot] = 0.5
    out_post = walkforward.run(panel_post, [baselines.Ridge()], oos_year=2020)

    pre = out_clean[out_clean["date"] <= pivot].sort_values(["date", "asset_id"]).reset_index(drop=True)
    post = out_post[out_post["date"] <= pivot].sort_values(["date", "asset_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(pre["forecast"], post["forecast"], check_exact=False, rtol=1e-10)
