"""
CLI entry point. Reads a YAML config, loads data, runs the harness, writes
long-format forecasts + summary table + plots (+ latency report).

Usage:
    python -m bench.run --config configs/replication_us_2022.yaml
    python -m bench.run --config configs/intraday_1m_smoke.yaml
"""

import argparse
import re
from pathlib import Path

# Import torch before pandas/NumPy on Windows. With recent CUDA wheels on RTX
# 50-series cards, importing pandas first can make torch fail while loading
# c10.dll. Only do the import at module load for CLI execution; tests and
# registry imports do not need to load the CUDA DLL stack.
_TORCH_PRELOAD = None
_TORCH_PRELOAD_ERROR: Exception | None = None
if __name__ == "__main__":
    try:
        import torch as _TORCH_PRELOAD
    except Exception as exc:
        _TORCH_PRELOAD_ERROR = exc

import pandas as pd
import yaml

from . import (
    baselines,
    metrics,
    report,
    walkforward,
)

MODEL_REGISTRY: dict[str, callable] = {
    "zero": baselines.Zero,
    "mean": baselines.Mean,
    "ridge": baselines.Ridge,
    "lightgbm": baselines.LightGBM,
    "catboost": baselines.CatBoost,
}

# Matches chronos-bolt-<ctx> or chronos-bolt-<ctx>-<cpu|cuda|auto>; same for chronos-2.
_CHRONOS_RE = re.compile(r"^(chronos-bolt|chronos-2)-(\d+)(?:-(cpu|cuda|auto))?$")
_TIMESFM_RE = re.compile(r"^(timesfm-2\.5(?:-xreg)?)-(\d+)$")
_MOIRAI_RE = re.compile(r"^moirai-1\.1-(small|base|large)-(\d+)(?:-(cpu|cuda|auto))?$")


def _uses_torch_cuda_model(model_names: list[str]) -> bool:
    return any(
        _CHRONOS_RE.match(name) or _TIMESFM_RE.match(name) or _MOIRAI_RE.match(name)
        for name in model_names
    )


def _preload_torch_if_needed(model_names: list[str]) -> None:
    """Load torch before compiled baseline libraries start fitting.

    On Windows, importing Torch after other native ML libraries have already
    trained can fail while loading torch DLLs. CUDA TSFM runs need Torch
    anyway, so make the DLL load order explicit and fail early with a useful
    environment summary.
    """
    if not _uses_torch_cuda_model(model_names):
        return
    global _TORCH_PRELOAD, _TORCH_PRELOAD_ERROR
    if _TORCH_PRELOAD is None and _TORCH_PRELOAD_ERROR is None:
        try:
            import torch as _TORCH_PRELOAD
        except Exception as exc:
            _TORCH_PRELOAD_ERROR = exc
    if _TORCH_PRELOAD is None:
        raise RuntimeError(
            "Torch failed to import before running TSFM models. Reinstall the "
            "CUDA-enabled torch wheel from requirements.txt."
        ) from _TORCH_PRELOAD_ERROR
    torch = _TORCH_PRELOAD
    print(
        "torch:",
        torch.__version__,
        "cuda:",
        torch.version.cuda,
        "available:",
        torch.cuda.is_available(),
        flush=True,
    )
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), flush=True)


def _build_chronos(name: str):
    m = _CHRONOS_RE.match(name)
    if not m:
        return None
    family, ctx, device = m.group(1), int(m.group(2)), m.group(3)
    if family == "chronos-bolt":
        from . import chronos_bolt
        if device is not None:
            return chronos_bolt.ChronosBolt(context_len=ctx, device=device)
        return chronos_bolt.ChronosBolt(context_len=ctx)
    from . import chronos_two
    if device is not None:
        return chronos_two.ChronosTwo(context_len=ctx, device=device)
    return chronos_two.ChronosTwo(context_len=ctx)


def _build_timesfm(name: str):
    m = _TIMESFM_RE.match(name)
    if not m:
        return None
    family, ctx = m.group(1), int(m.group(2))
    from . import timesfm_wrappers
    return timesfm_wrappers.TimesFM(context_len=ctx, use_xreg=family.endswith("-xreg"))


def _build_moirai(name: str):
    m = _MOIRAI_RE.match(name)
    if not m:
        return None
    size, ctx, device = m.group(1), int(m.group(2)), m.group(3)
    from . import moirai
    return moirai.Moirai(context_len=ctx, size=size, device=device or "auto")


def build_models(names: list[str]):
    out = []
    for n in names:
        if n in MODEL_REGISTRY:
            out.append(MODEL_REGISTRY[n]())
            continue
        m = _build_chronos(n)
        if m is not None:
            out.append(m)
            continue
        m = _build_timesfm(n)
        if m is not None:
            out.append(m)
            continue
        m = _build_moirai(n)
        if m is not None:
            out.append(m)
            continue
        raise KeyError(
            f"unknown model: {n}. registered: {sorted(MODEL_REGISTRY)}, "
            "or chronos-bolt-<ctx>[-cpu|-cuda|-auto], "
            "chronos-2-<ctx>[-cpu|-cuda|-auto], timesfm-2.5[-xreg]-<ctx>, "
            "or moirai-1.1-<small|base|large>-<ctx>[-cpu|-cuda|-auto]"
        )
    return out


def load_data(cfg: dict):
    src = cfg["source"]
    if src == "yfinance":
        from . import data_yfinance
        return data_yfinance.load_panel(
            start=cfg["start"], end=cfg["end"],
            cache_dir=cfg["cache_dir"],
            tickers=cfg.get("tickers"),
            top_n=cfg.get("top_n"),
        )
    if src == "ff25":
        from . import data_ff25
        return data_ff25.load_panel(
            start=cfg["start"], end=cfg["end"], cache_dir=cfg["cache_dir"],
        )
    if src == "wrds_panel":
        from . import data_wrds_panel
        return data_wrds_panel.load_panel(
            panel_path=cfg.get("panel_path", "src/data/datasets/panel.parquet"),
            start=cfg.get("start"),
            end=cfg.get("end"),
            top_n_by_mcap=cfg.get("top_n_by_mcap"),
            min_history_days=int(cfg.get("min_history_days", 252)),
            feature_set=cfg.get("feature_set", "core"),
            oos_year=cfg.get("oos_year"),
        )
    if src in ("yfinance_intraday", "synthetic_intraday", "wrds"):
        from . import data_intraday
        return data_intraday.load_panel(cfg)
    raise ValueError(f"unknown data source: {src}")


def _resolve_window(cfg: dict, panel) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return (train_end, oos_start, oos_end). Supports two config styles:

      oos_year: 2022                  # daily replication
      oos_start: 2024-03-15 09:30     # intraday explicit
      oos_end:   2024-03-15 16:00
      train_end: 2024-03-14 16:00     # optional; else min(panel) → oos_start-1bar
    """
    tz = panel.returns.index.tz

    def _norm(x) -> pd.Timestamp:
        t = pd.Timestamp(x)
        if tz is None:
            return t.tz_localize(None) if t.tzinfo is not None else t
        return t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)

    if "oos_year" in cfg:
        y = int(cfg["oos_year"])
        return (
            _norm(f"{y - 1}-12-31"),
            _norm(f"{y}-01-01"),
            _norm(f"{y}-12-31 23:59:59"),
        )
    oos_start = _norm(cfg["oos_start"])
    oos_end = _norm(cfg["oos_end"])
    if "train_end" in cfg:
        train_end = _norm(cfg["train_end"])
    else:
        all_dates = panel.returns.index
        prior = all_dates[all_dates < oos_start]
        if len(prior) == 0:
            raise ValueError("no in-sample bars precede oos_start; widen the data window")
        train_end = prior[-1]
    return train_end, oos_start, oos_end


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    _preload_torch_if_needed(cfg["models"])
    panel = load_data(cfg["data"])
    models = build_models(cfg["models"])
    horizon = int(cfg.get("horizon", 1))
    bars_per_year = int(cfg.get("bars_per_year", metrics.TRADING_DAYS))

    # Optional latency instrumentation.
    timed = None
    if cfg.get("latency", False):
        from . import latency
        timed = [latency.TimedForecaster(m) for m in models]
        models_to_run = timed
    else:
        models_to_run = models

    train_end, oos_start, oos_end = _resolve_window(cfg, panel)
    out = walkforward.run_window(
        panel, models_to_run,
        train_end=train_end, oos_start=oos_start, oos_end=oos_end,
        horizon=horizon,
    )

    out_dir = Path(cfg["output_dir"])
    report.write_long_csv(out, out_dir / "forecasts.csv")
    table = report.write_summary(out, out_dir / "summary", bars_per_year=bars_per_year)
    if not args.no_plots:
        from . import plots
        plots.write_all(out, out_dir, bars_per_year=bars_per_year)

    if timed is not None:
        from . import latency
        lat_df = latency.summarize([m.record for m in timed], bars_per_year=bars_per_year)
        latency.write_report(lat_df, out_dir)
        if not args.no_plots:
            latency.write_plots(timed, out_dir)
        print("\nLatency:")
        print(lat_df.to_string(float_format="{:.4f}".format))

    print(table.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
