"""
Chronos-Bolt zero-shot wrapper.

Univariate forecaster: each asset's return history is passed to Chronos-Bolt.
Model imports and weight loading are lazy so config parsing and tests do not
download model packages.
"""

from __future__ import annotations

import pandas as pd

from .protocols import Panel


class ChronosBolt:
    name = "chronos-bolt"

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        context_len: int = 252,
        device: str = "auto",
        dtype: str | None = None,
    ) -> None:
        self.name = f"chronos-bolt-{context_len}"
        self._model_id = model_id
        self._ctx = context_len
        self._device_requested = device
        self._dtype_requested = dtype
        self._pipeline = None
        self._torch = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        return None  # zero-shot

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        self._ensure_pipeline()
        end_idx = panel.returns.index.get_loc(asof) + 1
        start_idx = max(0, end_idx - self._ctx)
        window = panel.returns.iloc[start_idx:end_idx]

        valid = window.notna().all(axis=0)
        if not valid.any():
            return _nan_forecast(panel)
        ctx = self._torch.tensor(window.loc[:, valid].to_numpy().T, dtype=self._torch.float32)

        _, mean = self._pipeline.predict_quantiles(
            ctx, prediction_length=horizon, quantile_levels=[0.5]
        )
        yhat = mean[:, horizon - 1].float().cpu().numpy()

        out = pd.Series(yhat, index=window.columns[valid], dtype=float)
        return out.reindex(panel.assets())

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise ImportError(
                "Chronos-Bolt requires torch and chronos-forecasting. Install with "
                "`pip install -r requirements.txt`."
            ) from e

        device = _resolve_device(torch, self._device_requested)
        dtype = _resolve_dtype(torch, device, self._dtype_requested)
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self._model_id, device_map=device, dtype=dtype
        )
        self._torch = torch


def _resolve_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Chronos-Bolt requested cuda, but torch.cuda.is_available() is false")
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
