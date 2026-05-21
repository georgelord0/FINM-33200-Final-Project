"""
Chronos-2 zero-shot multivariate wrapper.

Same model family as ChronosBolt but consumes panel.covariates as
past_covariates. This is the actual project extension over the paper, which
only tested univariate TSFMs.
"""

import pandas as pd
import torch
from chronos import Chronos2Pipeline

from .protocols import Panel


class ChronosTwo:
    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        context_len: int = 252,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.name = f"chronos-2-{context_len}"
        self._ctx = context_len
        self._pipeline = Chronos2Pipeline.from_pretrained(
            model_id, device_map=device, dtype=dtype
        )

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        pass

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        end_idx = panel.returns.index.get_loc(asof) + 1
        start_idx = max(0, end_idx - self._ctx)
        window_dates = panel.returns.index[start_idx:end_idx]
        target_wide = panel.returns.iloc[start_idx:end_idx]

        if panel.covariates is None:
            cov_lookup = None
        else:
            dates_set = set(window_dates)
            mask = panel.covariates.index.get_level_values("date").isin(dates_set)
            cov_lookup = panel.covariates[mask]

        inputs = []
        valid_assets = []
        for asset in target_wide.columns:
            tgt = target_wide[asset]
            if tgt.isna().any():
                continue
            entry = {"target": torch.tensor(tgt.to_numpy(), dtype=torch.float32)}
            if cov_lookup is not None:
                try:
                    cov_a = cov_lookup.xs(asset, level="asset_id").reindex(window_dates)
                except KeyError:
                    continue
                if cov_a.isna().any().any():
                    continue
                entry["past_covariates"] = {
                    name: torch.tensor(cov_a[name].to_numpy(), dtype=torch.float32)
                    for name in cov_a.columns
                }
            inputs.append(entry)
            valid_assets.append(asset)

        if not inputs:
            return pd.Series(0.0, index=panel.assets())

        _, means = self._pipeline.predict_quantiles(
            inputs, prediction_length=horizon, quantile_levels=[0.5]
        )
        yhat = [m[0, horizon - 1].float().item() for m in means]
        out = pd.Series(yhat, index=valid_assets, dtype=float)
        return out.reindex(panel.assets()).fillna(0.0)
