"""
CLI entry point. Reads a YAML config, loads real data, runs the harness,
writes long-format forecasts + summary table + plots.

Usage:
    python -m bench.run --config configs/replication_us_2022.yaml
"""

import argparse
from pathlib import Path

import yaml

from . import baselines, data_ff25, data_yfinance, plots, report, walkforward

MODEL_REGISTRY: dict[str, callable] = {
    "zero": baselines.Zero,
    "mean": baselines.Mean,
    "ridge": baselines.Ridge,
    "lightgbm": baselines.LightGBM,
    "catboost": baselines.CatBoost,
}

# TSFM wrappers pulled in only on demand (heavy torch import).
def _register_tsfm() -> None:
    try:
        from . import chronos_bolt
        for ctx in (21, 252):
            MODEL_REGISTRY[f"chronos-bolt-{ctx}"] = (
                lambda c=ctx: chronos_bolt.ChronosBolt(context_len=c)
            )
    except ImportError:
        pass
    try:
        from . import chronos_two
        for ctx in (21, 252):
            MODEL_REGISTRY[f"chronos-2-{ctx}"] = (
                lambda c=ctx: chronos_two.ChronosTwo(context_len=c)
            )
    except ImportError:
        pass


def build_models(names: list[str]):
    if any(n.startswith("chronos") or n.startswith("timesfm") or n.startswith("moirai")
           for n in names):
        _register_tsfm()
    out = []
    for n in names:
        if n not in MODEL_REGISTRY:
            raise KeyError(f"unknown model: {n}. registered: {sorted(MODEL_REGISTRY)}")
        out.append(MODEL_REGISTRY[n]())
    return out


def load_data(cfg: dict):
    src = cfg["source"]
    if src == "yfinance":
        return data_yfinance.load_panel(
            start=cfg["start"], end=cfg["end"],
            cache_dir=cfg["cache_dir"],
            tickers=cfg.get("tickers"),
            top_n=cfg.get("top_n"),
        )
    if src == "ff25":
        return data_ff25.load_panel(
            start=cfg["start"], end=cfg["end"], cache_dir=cfg["cache_dir"],
        )
    raise ValueError(f"unknown data source: {src}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    panel = load_data(cfg["data"])
    models = build_models(cfg["models"])
    out = walkforward.run(panel, models, oos_year=int(cfg["oos_year"]),
                          horizon=int(cfg.get("horizon", 1)))

    out_dir = Path(cfg["output_dir"])
    report.write_long_csv(out, out_dir / "forecasts.csv")
    table = report.write_summary(out, out_dir / "summary")
    if not args.no_plots:
        plots.write_all(out, out_dir)

    print(table.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
