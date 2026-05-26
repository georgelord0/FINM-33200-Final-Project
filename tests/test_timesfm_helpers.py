"""Pure-numpy tests for the xreg future-covariate helper. No timesfm needed."""

import numpy as np

from bench.timesfm_wrappers import _future_covariate


def test_lagged_return_future_uses_target_history():
    hist = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    target = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    future = _future_covariate("ret_lag1", hist, target, horizon=1)
    np.testing.assert_allclose(future, [target[-1]])


def test_ret_lag5_future_value_walks_back_five_steps():
    hist = np.zeros(10)
    target = np.arange(10, dtype=float)
    future = _future_covariate("ret_lag5", hist, target, horizon=1)
    np.testing.assert_allclose(future, [target[-5]])


def test_non_lag_covariate_carries_forward():
    hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.zeros_like(hist)
    future = _future_covariate("vol_21", hist, target, horizon=3)
    np.testing.assert_allclose(future, [5.0, 5.0, 5.0])


def test_horizon_beyond_lag_falls_back_to_carry_forward():
    # ret_lag1 only known for h=1; h>=2 falls back to carry-forward
    hist = np.array([0.1, 0.2, 0.3])
    target = np.array([0.4, 0.5, 0.6])
    future = _future_covariate("ret_lag1", hist, target, horizon=3)
    assert future[0] == target[-1]
    assert future[1] == hist[-1]
    assert future[2] == hist[-1]
