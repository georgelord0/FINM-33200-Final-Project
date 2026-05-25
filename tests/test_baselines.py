import pandas as pd

from bench.baselines import _aligned_xy
from bench.protocols import Panel


def test_aligned_xy_keeps_rows_with_missing_covariates() -> None:
    dates = pd.bdate_range("2021-12-27", periods=5)
    returns = pd.DataFrame(
        {
            1: [0.01, 0.02, 0.03, 0.04, 0.05],
            2: [0.02, 0.01, 0.00, -0.01, -0.02],
        },
        index=dates,
    )
    returns.index.name = "date"
    returns.columns.name = "asset_id"

    idx = pd.MultiIndex.from_product([dates, [1, 2]], names=["date", "asset_id"])
    covariates = pd.DataFrame(
        {
            "usable": range(len(idx)),
            "sometimes_missing": [None, 1.0] * 5,
            "always_missing": [None] * len(idx),
        },
        index=idx,
    )

    x, y = _aligned_xy(
        Panel(returns=returns, covariates=covariates, freq="B"),
        train_end=pd.Timestamp("2021-12-31"),
        horizon=1,
    )

    assert len(x) == 8
    assert len(y) == 8
    assert not x.isna().any().any()
