"""Tabular outputs: long forecast CSV + per-model summary markdown table."""

from pathlib import Path

import pandas as pd

from . import metrics


def write_long_csv(out_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)


def summary_table(out_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in out_df.groupby("model", sort=False):
        s = metrics.summarize(group)
        s["model"] = name
        rows.append(s)
    df = pd.DataFrame(rows).set_index("model")
    cols = ["r2_oos", "mae", "rmse", "dir_acc", "f1",
            "ann_return", "ann_vol", "sharpe", "max_dd", "max_dd_1d",
            "n_obs", "n_dates"]
    return df[cols]


def write_summary(out_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    tab = summary_table(out_df)
    path.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(path.with_suffix(".csv"))
    with path.with_suffix(".md").open("w") as f:
        f.write(tab.to_markdown(floatfmt=".4f"))
        f.write("\n")
    return tab
