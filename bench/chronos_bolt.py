"""
Chronos-Bolt zero-shot wrapper.

Univariate forecaster: takes the last `context_len` returns per asset and
asks Chronos-Bolt for a one-step quantile forecast. We use the model's mean
prediction as the point forecast (matches the paper's comparison setup).

This is the same family of model the Rahimikia/Ni/Wang 2025 paper benchmarks
in its zero-shot regime, so it's our anchor for the replication.
"""

import pandas as pd
import torch
from chronos import BaseChronosPipeline

from .protocols import Panel


class ChronosBolt:
    name = "chronos-bolt"

    def __init__(
        self,
        model_id: str = "amazon/chronos-bolt-base",
        context_len: int = 252,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.name = f"chronos-bolt-{context_len}"
        self._ctx = context_len
        self._device = device
        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_id, device_map=device, dtype=dtype
        )

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        pass  # zero-shot

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        end_idx = panel.returns.index.get_loc(asof) + 1
        start_idx = max(0, end_idx - self._ctx)
        window = panel.returns.iloc[start_idx:end_idx]

        # Keep only assets with the full context window of finite values.
        valid = window.notna().all(axis=0)
        if not valid.any():
            return pd.Series(0.0, index=panel.assets())
        ctx = torch.tensor(window.loc[:, valid].to_numpy().T, dtype=torch.float32)

        _, mean = self._pipeline.predict_quantiles(
            ctx, prediction_length=horizon, quantile_levels=[0.5]
        )
        yhat = mean[:, horizon - 1].float().cpu().numpy()

        out = pd.Series(yhat, index=window.columns[valid])
        return out.reindex(panel.assets()).fillna(0.0)
