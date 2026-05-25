"""
Salesforce MOIRAI zero-shot wrapper via Uni2TS/GluonTS.

This project uses the stable target-only GluonTS inference path. The report
labels MOIRAI as target-only unless a future Uni2TS covariate adapter is added.
"""

from __future__ import annotations

import pandas as pd

from .protocols import Panel


class Moirai:
    def __init__(
        self,
        context_len: int = 252,
        *,
        size: str = "small",
        model_id: str | None = None,
        device: str = "auto",
        batch_size: int = 32,
        num_samples: int = 100,
    ) -> None:
        self.name = f"moirai-1.1-{size}-{context_len}"
        self._ctx = context_len
        self._size = size
        self._model_id = model_id or f"Salesforce/moirai-1.1-R-{size}"
        self._device = device
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._predictor = None
        self._prediction_length = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        return None

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        self._ensure_predictor(horizon)
        end_idx = panel.returns.index.get_loc(asof) + 1
        start_idx = max(0, end_idx - self._ctx)
        window = panel.returns.iloc[start_idx:end_idx]

        valid = window.notna().all(axis=0)
        if not valid.any():
            return pd.Series(float("nan"), index=panel.assets(), dtype=float)

        valid_window = window.loc[:, valid]
        ds = self._pandas_dataset(dict(valid_window))
        forecasts = list(self._predictor.predict(ds))
        yhat = [float(fc.mean[horizon - 1]) for fc in forecasts]
        out = pd.Series(yhat, index=valid_window.columns, dtype=float)
        return out.reindex(panel.assets())

    def _ensure_predictor(self, prediction_length: int) -> None:
        if self._predictor is not None and self._prediction_length == prediction_length:
            return
        try:
            import torch
            from gluonts.dataset.pandas import PandasDataset
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as e:
            raise ImportError(
                "MOIRAI requires torch, gluonts, and uni2ts. Install with "
                "`pip install -r requirements.txt`."
            ) from e

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MOIRAI requested cuda, but torch.cuda.is_available() is false")
        else:
            device = self._device

        module = MoiraiModule.from_pretrained(self._model_id)
        try:
            module = module.to(device)
        except AttributeError:
            pass
        model = MoiraiForecast(
            module=module,
            prediction_length=prediction_length,
            context_length=self._ctx,
            patch_size="auto",
            num_samples=self._num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._pandas_dataset = PandasDataset
        self._predictor = model.create_predictor(batch_size=self._batch_size)
        self._prediction_length = prediction_length
