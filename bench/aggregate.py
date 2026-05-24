"""
Combine per-run summary CSVs into one cross-run comparison table.

Usage:
    python -m bench.aggregate <label1>:<dir1> <label2>:<dir2> ...
        [--out results/combined.md]

Each <dir> must contain a summary.csv written by bench.report.
"""

import argparse
from pathlib import Path

import pandas as pd

METRICS = ["r2_oos", "dir_acc", "sharpe", "ann_return", "max_dd"]


def combine(runs: dict[str, Path]) -> pd.DataFrame:
    parts = []
    for label, d in runs.items():
        df = pd.read_csv(d / "summary.csv").set_index("model")[METRICS]
        df.columns = pd.MultiIndex.from_product([[label], df.columns])
        parts.append(df)
    return pd.concat(parts, axis=1).sort_index(axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label:path entries, e.g. 2022:results/replication_us_2022")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    parsed = {}
    for entry in args.runs:
        label, _, path = entry.partition(":")
        parsed[label] = Path(path)

    table = combine(parsed)
    md = table.to_markdown(floatfmt=".4f")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
