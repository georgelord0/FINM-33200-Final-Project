"""
TimesFM 2.5 zero-shot wrappers.

Two modes are exposed through the same class:
  * target-only TimesFM forecast
  * XReg forecast with numeric dynamic covariates

For the one-step daily equity setup, XReg future covariates are carried forward
from the last known as-of value because same-day features are known after close.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .protocols import Panel

# sanity clamp on xreg output — saw forecasts of ±1 on intraday before
_XREG_FORECAST_CLAMP = 0.20

_RET_LAG_RE = re.compile(r"^ret_lag(\d+)$")


class TimesFM:
    def __init__(
        self,
        context_len: int = 252,
        *,
        use_xreg: bool = False,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        xreg_mode: str = "xreg + timesfm",
        batch_size: int = 32,
    ) -> None:
        self.name = f"timesfm-2.5{'-xreg' if use_xreg else ''}-{context_len}"
        self._ctx = context_len
        self._use_xreg = use_xreg
        self._model_id = model_id
        self._xreg_mode = xreg_mode
        self._batch_size = batch_size
        self._model = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        return None

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        self._ensure_model(max_horizon=horizon)
        end_idx = panel.returns.index.get_loc(asof) + 1
        start_idx = max(0, end_idx - self._ctx)
        window = panel.returns.iloc[start_idx:end_idx]

        if self._use_xreg:
            return self._predict_xreg(panel, window, horizon)
        return self._predict_univariate(panel, window, horizon)

    def _predict_univariate(
        self, panel: Panel, window: pd.DataFrame, horizon: int
    ) -> pd.Series:
        valid = window.notna().all(axis=0)
        if not valid.any():
            return _nan_forecast(panel)
        assets = list(window.columns[valid])
        inputs = [window[a].to_numpy(dtype=float) for a in assets]
        point, _ = self._model.forecast(horizon=horizon, inputs=inputs)
        out = pd.Series(np.asarray(point)[:, horizon - 1], index=assets, dtype=float)
        return out.reindex(panel.assets())

    def _predict_xreg(
        self, panel: Panel, window: pd.DataFrame, horizon: int
    ) -> pd.Series:
        if panel.covariates is None:
            return self._predict_univariate(panel, window, horizon)

        window_dates = window.index
        dates = panel.covariates.index.get_level_values("date")
        cov_lookup = panel.covariates[dates.isin(window_dates)]

        assets: list = []
        inputs: list[np.ndarray] = []
        covariates: dict[str, list[np.ndarray]] = {
            name: [] for name in panel.covariates.columns
        }

        for asset in window.columns:
            target = window[asset]
            if target.isna().any():
                continue
            try:
                cov_a = cov_lookup.xs(asset, level="asset_id").reindex(window_dates)
            except KeyError:
                continue
            cov_a = cov_a.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

            target_arr = target.to_numpy(dtype=float)
            inputs.append(target_arr)
            assets.append(asset)
            for name in covariates:
                hist = cov_a[name].to_numpy(dtype=float)
                covariates[name].append(
                    np.concatenate([hist, _future_covariate(name, hist, target_arr, horizon)])
                )

        if not inputs:
            return _nan_forecast(panel)

        point, _ = self._model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates=covariates,
            dynamic_categorical_covariates={},
            static_categorical_covariates={},
            xreg_mode=self._xreg_mode,
        )
        raw = np.asarray(point)[:, horizon - 1]
        out = pd.Series(raw, index=assets, dtype=float).clip(
            -_XREG_FORECAST_CLAMP, _XREG_FORECAST_CLAMP
        )
        return out.reindex(panel.assets())

    def _ensure_model(self, max_horizon: int) -> None:
        if self._model is not None:
            return
        try:
            import timesfm
        except ImportError as e:
            raise ImportError(
                "TimesFM wrappers require timesfm with XReg extras. Install with "
                "`pip install -r requirements.txt`."
            ) from e

        model_cls = timesfm.TimesFM_2p5_200M_torch
        try:
            model = model_cls.from_pretrained(self._model_id)
        except TypeError as exc:
            if "proxies" not in str(exc):
                raise
            # huggingface_hub>=0.36 passes `proxies` through HubMixin in a way
            # this TimesFM class does not accept. Call the backend loader
            # directly with the same defaults, excluding the stray kwarg.
            model = model_cls._from_pretrained(
                model_id=self._model_id,
                revision=None,
                cache_dir=None,
                force_download=False,
                local_files_only=False,
                token=None,
            )
        model.compile(
            timesfm.ForecastConfig(
                max_context=self._ctx,
                max_horizon=max_horizon,
                normalize_inputs=True,
                per_core_batch_size=self._batch_size,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
                return_backcast=self._use_xreg,
            )
        )
        self._model = model


def _future_covariate(name: str, hist: np.ndarray, target: np.ndarray, horizon: int) -> np.ndarray:
    """Future values for an xreg covariate over the horizon.

    For ret_lag<N> the future is deterministic from the target history:
    ret_lag<N>[t+h] = target[t+h-N] when h <= N. Old code carried forward
    hist[-1] for every column, which made ret_lag1[t+1] = yesterday's return
    when it should be today's. Slow features (vol, momentum) still carry
    forward.
    """
    m = _RET_LAG_RE.match(name)
    if m is None:
        return np.repeat(hist[-1], horizon)
    n_lag = int(m.group(1))
    future = []
    for h in range(1, horizon + 1):
        idx = h - n_lag
        if idx <= 0 and abs(idx) < len(target):
            future.append(target[idx - 1])
        else:
            future.append(hist[-1])
    return np.asarray(future, dtype=float)


def _nan_forecast(panel: Panel) -> pd.Series:
    return pd.Series(float("nan"), index=panel.assets(), dtype=float)
