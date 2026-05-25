"""
Chronos-2 zero-shot covariate-aware wrapper.

Chronos-2 consumes each asset's target history plus the matching past
covariate history. Future covariates are not required for the one-step-ahead
setup used in this project.
"""

from __future__ import annotations

import pandas as pd

from .protocols import Panel


class ChronosTwo:
    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        context_len: int = 252,
        device: str = "auto",
        dtype: str | None = None,
    ) -> None:
        self.name = f"chronos-2-{context_len}"
        self._model_id = model_id
        self._ctx = context_len
        self._device_requested = device
        self._dtype_requested = dtype
        self._pipeline = None
        self._torch = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        return None

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        self._ensure_pipeline()
        end_idx = panel.returns.index.get_loc(asof) + 1
        start_idx = max(0, end_idx - self._ctx)
        window_dates = panel.returns.index[start_idx:end_idx]
        target_wide = panel.returns.iloc[start_idx:end_idx]

        cov_lookup = None
        if panel.covariates is not None:
            dates = panel.covariates.index.get_level_values("date")
            cov_lookup = panel.covariates[dates.isin(window_dates)]

        inputs = []
        valid_assets = []
        for asset in target_wide.columns:
            tgt = target_wide[asset]
            if tgt.isna().any():
                continue
            entry = {"target": self._torch.tensor(tgt.to_numpy(), dtype=self._torch.float32)}
            if cov_lookup is not None:
                try:
                    cov_a = cov_lookup.xs(asset, level="asset_id").reindex(window_dates)
                except KeyError:
                    continue
                if cov_a.isna().any().any():
                    continue
                entry["past_covariates"] = {
                    name: self._torch.tensor(cov_a[name].to_numpy(), dtype=self._torch.float32)
                    for name in cov_a.columns
                }
            inputs.append(entry)
            valid_assets.append(asset)

        if not inputs:
            return _nan_forecast(panel)

        _, means = self._pipeline.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=[0.5]
        )
        yhat = [m[0, horizon - 1].float().item() for m in means]
        out = pd.Series(yhat, index=valid_assets, dtype=float)
        return out.reindex(panel.assets())

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError(
                "Chronos-2 requires torch and chronos-forecasting. Install with "
                "`pip install -r requirements.txt`."
            ) from e
        device = _resolve_device(torch, self._device_requested)
        dtype = _resolve_dtype(torch, device, self._dtype_requested)
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self._model_id, device_map=device, dtype=dtype
        )
        self._torch = torch


def _resolve_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Chronos-2 requested cuda, but torch.cuda.is_available() is false")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return device


def _resolve_dtype(torch, device: str, dtype: str | None):
    if dtype is None:
        return torch.bfloat16 if device == "cuda" else torch.float32
    try:
        return getattr(torch, dtype)
    except AttributeError as e:
        raise ValueError(f"unknown torch dtype: {dtype}") from e


def _nan_forecast(panel: Panel) -> pd.Series:
    return pd.Series(float("nan"), index=panel.assets(), dtype=float)
