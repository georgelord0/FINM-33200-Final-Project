"""
Latency instrumentation for the benchmark harness.

`TimedForecaster` wraps any Forecaster and records wall-clock for fit and
each predict call. Drops into `walkforward.run_window` because it satisfies
the same Protocol.

Timings use `time.perf_counter()` (monotonic, ns-resolution). For Chronos /
TimesFM / MOIRAI wrappers running on CUDA we call `torch.cuda.synchronize()`
before stopping the clock so the timer doesn't race the kernel launch. The
torch import is lazy — CPU-only runs don't need torch installed.

Outputs (written by `write_report`):
    latency.csv          per-model row: fit_seconds, predict_mean,
                         p50/p95/p99, assets_per_sec, bars_per_sec
    latency.md           same, markdown
    latency_p95.png      bar chart of predict p95 per model
    latency_throughput.png  scatter of predict_seconds vs assets_per_call
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .protocols import Forecaster, Panel


def _torch_sync() -> None:
    """Sync CUDA before stopping the clock. No-op if torch missing / no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@dataclass
class LatencyRecord:
    model: str
    fit_seconds: float = 0.0
    predict_seconds: list[float] = field(default_factory=list)
    n_assets_per_call: list[int] = field(default_factory=list)


class TimedForecaster:
    """Forecaster Protocol-compatible decorator. Records timings on `record`.

    Use the inner model's `name` so downstream reporting (summary.csv, plots)
    sees the same identifier whether or not latency is enabled.
    """

    def __init__(self, inner: Forecaster) -> None:
        self._inner = inner
        self.name = inner.name
        self.record = LatencyRecord(model=inner.name)

    def fit(self, panel: Panel, train_end: pd.Timestamp) -> None:
        t0 = time.perf_counter()
        self._inner.fit(panel, train_end)
        _torch_sync()
        self.record.fit_seconds = time.perf_counter() - t0

    def predict(self, panel: Panel, asof: pd.Timestamp, horizon: int = 1) -> pd.Series:
        t0 = time.perf_counter()
        out = self._inner.predict(panel, asof, horizon=horizon)
        _torch_sync()
        self.record.predict_seconds.append(time.perf_counter() - t0)
        self.record.n_assets_per_call.append(int(out.notna().sum()))
        return out


def summarize(records: list[LatencyRecord], bars_per_year: int) -> pd.DataFrame:
    """Per-model summary table. bars_per_year is informational — bars_per_sec
    is the observed call rate, not annualised."""
    rows = []
    for r in records:
        if not r.predict_seconds:
            rows.append({"model": r.model, "fit_seconds": r.fit_seconds,
                         "n_calls": 0, "predict_mean": float("nan"),
                         "p50": float("nan"), "p95": float("nan"), "p99": float("nan"),
                         "assets_per_sec": float("nan"), "bars_per_sec": float("nan"),
                         "total_predict_seconds": 0.0})
            continue
        ps = np.array(r.predict_seconds)
        na = np.array(r.n_assets_per_call)
        total = ps.sum()
        rows.append({
            "model": r.model,
            "fit_seconds": r.fit_seconds,
            "n_calls": len(ps),
            "predict_mean": float(ps.mean()),
            "p50": float(np.percentile(ps, 50)),
            "p95": float(np.percentile(ps, 95)),
            "p99": float(np.percentile(ps, 99)),
            "assets_per_sec": float(na.sum() / total) if total > 0 else float("nan"),
            "bars_per_sec": float(len(ps) / total) if total > 0 else float("nan"),
            "total_predict_seconds": float(total),
        })
    return pd.DataFrame(rows).set_index("model")


def write_report(lat_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat_df.to_csv(out_dir / "latency.csv")
    with (out_dir / "latency.md").open("w") as f:
        f.write(lat_df.to_markdown(floatfmt=".4f"))
        f.write("\n")


def write_plots(timed: list[TimedForecaster], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # p95 bar chart.
    p95 = {t.name: (np.percentile(t.record.predict_seconds, 95)
                    if t.record.predict_seconds else float("nan"))
           for t in timed}
    s = pd.Series(p95).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(s.index, s.values, color="0.3")
    ax.set_xlabel("predict() p95 (seconds)")
    ax.set_title("Per-model predict latency, p95")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_p95.png", dpi=150)
    plt.close(fig)

    # Throughput scatter: seconds vs assets-per-call, one trace per model.
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in timed:
        if not t.record.predict_seconds:
            continue
        ax.scatter(t.record.n_assets_per_call, t.record.predict_seconds,
                   s=8, alpha=0.6, label=t.name)
    ax.set_xlabel("assets per predict() call")
    ax.set_ylabel("seconds per call")
    ax.set_title("Latency vs panel width")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / "latency_throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
