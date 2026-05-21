"""
Baseline forecasters: Zero, Mean, Ridge, LightGBM, CatBoost.

Ridge, LightGBM, CatBoost are pooled cross-sectional models trained on the
panel's covariates. They share an _aligned_xy helper. Zero and Mean are
stateless sanity baselines. Zero exists so the harness can prove it
doesn't lie (R^2_oos must equal 0 by construction).
"""

import lightgbm as lgb
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge as SKRidge

from .protocols import Panel


def _aligned_xy(panel: Panel, train_end: pd.Timestamp, horizon: int):
    """Build (features, target) for training.

    A training row indexed (t, asset) has target = returns[t+horizon, asset],
    so we must restrict to t such that t+horizon (in trading days) is itself
    <= train_end. Filtering on t alone would slip `horizon` rows of OOS-year
    returns into the training labels.
    """
    if panel.covariates is None:
        raise ValueError("ridge/lightgbm/catboost baselines require covariates")
    target = panel.returns.shift(-horizon).stack().rename("target")
    target.index.names = ["date", "asset_id"]
    df = panel.covariates.join(target, how="inner").dropna()

    idx = panel.returns.index
    cutoff_pos = idx.searchsorted(train_end, side="right") - horizon - 1
    if cutoff_pos < 0:
        df = df.iloc[:0]
    else:
        df = df[df.index.get_level_values("date") <= idx[cutoff_pos]]
    return df.drop(columns=["target"]), df["target"]


def _features_at(panel: Panel, asof: pd.Timestamp) -> pd.DataFrame | None:
    try:
        return panel.covariates.xs(asof, level="date")
    except KeyError:
        return None


class Zero:
    name = "zero"

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        pass

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        return pd.Series(0.0, index=panel.assets())


class Mean:
    name = "mean"

    def __init__(self) -> None:
        self._mu: pd.Series | None = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        self._mu = panel.returns.loc[:train_end].mean()

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        return self._mu.reindex(panel.assets()).fillna(0.0)


class Ridge:
    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        self._model = SKRidge(alpha=alpha)
        self._cols: list[str] | None = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        x, y = _aligned_xy(panel, train_end, horizon=1)
        self._cols = list(x.columns)
        self._model.fit(x.to_numpy(), y.to_numpy())

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        x = _features_at(panel, asof)
        if x is None or x.empty:
            return pd.Series(0.0, index=panel.assets())
        x = x.reindex(columns=self._cols).fillna(0.0)
        yhat = self._model.predict(x.to_numpy())
        return pd.Series(yhat, index=x.index).reindex(panel.assets()).fillna(0.0)


class LightGBM:
    name = "lightgbm"

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05,
                 num_leaves: int = 31) -> None:
        self._params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                            num_leaves=num_leaves, verbose=-1)
        self._model: lgb.LGBMRegressor | None = None
        self._cols: list[str] | None = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        x, y = _aligned_xy(panel, train_end, horizon=1)
        self._cols = list(x.columns)
        self._model = lgb.LGBMRegressor(**self._params)
        self._model.fit(x, y)

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        x = _features_at(panel, asof)
        if x is None or x.empty:
            return pd.Series(0.0, index=panel.assets())
        x = x.reindex(columns=self._cols).fillna(0.0)
        yhat = self._model.predict(x)
        return pd.Series(yhat, index=x.index).reindex(panel.assets()).fillna(0.0)


class CatBoost:
    name = "catboost"

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05,
                 depth: int = 6) -> None:
        self._params = dict(iterations=iterations, learning_rate=learning_rate,
                            depth=depth, verbose=False, allow_writing_files=False)
        self._model: CatBoostRegressor | None = None
        self._cols: list[str] | None = None

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        x, y = _aligned_xy(panel, train_end, horizon=1)
        self._cols = list(x.columns)
        self._model = CatBoostRegressor(**self._params)
        self._model.fit(x, y)

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        x = _features_at(panel, asof)
        if x is None or x.empty:
            return pd.Series(0.0, index=panel.assets())
        x = x.reindex(columns=self._cols).fillna(0.0)
        yhat = self._model.predict(x)
        return pd.Series(yhat, index=x.index).reindex(panel.assets()).fillna(0.0)
