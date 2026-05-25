import pandas as pd
import pytest

from bench import data_wrds_panel


def _tiny_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-12-21", periods=8)
    rows = []
    for permno, mcap in [(1, 100.0), (2, 300.0), (3, 200.0)]:
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "permno": permno,
                "excess_ret": 0.001 * (permno + i),
                "target_ret": 9.0,
                "direction": 1,
                "prc": 50.0,
                "market_cap": mcap + i,
                "ret_lag_1": 0.01 * i,
                "mom_1m": 0.02 * i,
            })
    return pd.DataFrame(rows)


def test_wrds_panel_pivots_and_filters_features(tmp_path):
    path = tmp_path / "panel.parquet"
    _tiny_panel().to_parquet(path, index=False)

    panel = data_wrds_panel.load_panel(
        panel_path=path,
        start="2020-12-21",
        end="2020-12-31",
        top_n_by_mcap=2,
        min_history_days=5,
        feature_set="core",
        oos_year=2021,
    )

    assert list(panel.returns.columns) == [2, 3]
    assert panel.returns.shape == (8, 2)
    assert "ret_lag_1" in panel.covariates.columns
    assert "mom_1m" in panel.covariates.columns
    assert "target_ret" not in panel.covariates.columns
    assert "direction" not in panel.covariates.columns
    assert "prc" not in panel.covariates.columns


def test_wrds_panel_uses_excess_ret_not_forward_target(tmp_path):
    path = tmp_path / "panel.parquet"
    df = _tiny_panel()
    df.loc[df["permno"] == 1, "target_ret"] = 99.0
    df.to_parquet(path, index=False)

    panel = data_wrds_panel.load_panel(
        panel_path=path,
        top_n_by_mcap=1,
        min_history_days=5,
        feature_set=["ret_lag_1"],
        oos_year=2021,
    )

    assert panel.returns.iloc[0, 0] != 99.0


def test_wrds_panel_missing_file_errors_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="WRDS panel not found"):
        data_wrds_panel.load_panel(panel_path=tmp_path / "missing.parquet")
