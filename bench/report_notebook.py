"""Notebook-facing helpers for the generated project report.

The report notebook intentionally keeps code cells short. Data loading,
runtime cleanup, model coverage checks, and image display live here so the
notebook reads like a report rather than a script.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    title: str
    config_path: Path
    result_dir: Path
    description: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "wrds_daily_2022": BenchmarkSpec(
        key="wrds_daily_2022",
        title="Benchmark 1: WRDS Daily 2022",
        config_path=ROOT / "configs" / "wrds_daily_2022_full.yaml",
        result_dir=RESULTS / "wrds_daily_2022_full",
        description=(
            "Daily CRSP/Compustat/Fama-French panel, top 100 stocks by market "
            "capitalization, trained through 2021 and evaluated in 2022."
        ),
    ),
    "wrds_daily_2023": BenchmarkSpec(
        key="wrds_daily_2023",
        title="Benchmark 2: WRDS Daily 2023",
        config_path=ROOT / "configs" / "wrds_daily_2023_full.yaml",
        result_dir=RESULTS / "wrds_daily_2023_full",
        description=(
            "Same WRDS daily design, trained through 2022 and evaluated in 2023."
        ),
    ),
    "ff25_2023": BenchmarkSpec(
        key="ff25_2023",
        title="Benchmark 3: Fama-French 25 Daily 2023",
        config_path=ROOT / "configs" / "ff25_2023.yaml",
        result_dir=RESULTS / "ff25_2023",
        description=(
            "Public daily Fama-French 25 size/book-to-market portfolios, used as "
            "an external sanity benchmark that does not require WRDS credentials."
        ),
    ),
    "intraday_1m_wrds": BenchmarkSpec(
        key="intraday_1m_wrds",
        title="Benchmark 4: WRDS TAQ Intraday 1m",
        config_path=ROOT / "configs" / "intraday_1m_wrds.yaml",
        result_dir=RESULTS / "intraday_1m_wrds",
        description=(
            "WRDS TAQ one-minute bars for ten large-cap stocks. This run is the "
            "short-horizon extension and records latency for every configured model."
        ),
    ),
}


def display_runtime_environment(path: str | Path | None = None) -> None:
    """Render a concise runtime environment summary and package table."""
    info_path = Path(path) if path is not None else RESULTS / "system_info.json"
    display, Markdown = _display_tools()
    if not info_path.exists():
        display(Markdown(f"System info not found at `{info_path}`. Run `python scripts/reproduce.py --stage report`."))
        return

    info = json.loads(info_path.read_text(encoding="utf-8"))
    display(_runtime_summary_table(info))
    packages = _package_table(info)
    if not packages.empty:
        display(Markdown("#### Key Package Versions"))
        display(packages)


def display_cross_benchmark_summary() -> None:
    """Render one compact table across the four default benchmark outputs."""
    display, Markdown = _display_tools()
    frames = []
    for spec in BENCHMARKS.values():
        summary = load_summary(spec)
        if summary.empty:
            continue
        view = _summary_view(summary)
        view.insert(0, "benchmark", spec.title.replace("Benchmark ", "B"))
        frames.append(view)

    if not frames:
        display(Markdown("No benchmark summaries found yet. Run `python scripts/reproduce.py --stage bench`."))
        return

    display(pd.concat(frames, ignore_index=True))


def display_benchmark(key: str) -> None:
    """Render one benchmark section: metadata, model coverage, tables, plots."""
    display, Markdown = _display_tools()
    spec = BENCHMARKS[key]
    cfg = load_config(spec)
    summary = load_summary(spec)
    latency = load_latency(spec)

    display(Markdown(spec.description))
    display(_benchmark_metadata(spec, cfg))
    display(Markdown("#### Configured Model Coverage"))
    display(_model_coverage_table(cfg, summary))

    if summary.empty:
        display(Markdown(f"No summary found at `{spec.result_dir / 'summary.csv'}`. Run this config and regenerate the report."))
        return

    missing = missing_configured_models(cfg, summary)
    if missing:
        display(Markdown(
            "**Artifact note:** the current result files are missing configured "
            f"models: `{', '.join(missing)}`. Rerun `{_relpath(spec.config_path)}` "
            "and regenerate the report to populate these rows."
        ))

    display(Markdown("#### Accuracy and Portfolio Metrics"))
    display(_summary_view(summary))

    if cfg.get("latency") or not latency.empty:
        display(Markdown("#### Latency"))
        if latency.empty:
            display(Markdown("No latency file found yet for this latency-enabled config."))
        else:
            latency_missing = missing_configured_models(cfg, latency)
            if latency_missing:
                display(Markdown(
                    "**Latency artifact note:** the current latency file is missing "
                    f"configured models: `{', '.join(latency_missing)}`."
                ))
            display(_latency_view(latency))

    display(Markdown("#### Plots"))
    _display_images(
        spec,
        [
            "sharpe_by_model.png",
            "equity_curves.png",
            "latency_p95.png",
            "latency_throughput.png",
        ],
    )


def load_config(spec: BenchmarkSpec) -> dict[str, Any]:
    return yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))


def load_summary(spec: BenchmarkSpec) -> pd.DataFrame:
    return _load_csv_with_model(spec.result_dir / "summary.csv")


def load_latency(spec: BenchmarkSpec) -> pd.DataFrame:
    return _load_csv_with_model(spec.result_dir / "latency.csv")


def configured_result_models(cfg: dict[str, Any]) -> list[str]:
    return [_result_model_name(str(model)) for model in cfg.get("models", [])]


def missing_configured_models(cfg: dict[str, Any], artifact: pd.DataFrame) -> list[str]:
    if artifact.empty or "model" not in artifact.columns:
        return configured_result_models(cfg)
    actual = set(artifact["model"].astype(str))
    return [model for model in configured_result_models(cfg) if model not in actual]


def _runtime_summary_table(info: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("Run timestamp", info.get("collected_at_utc")),
        ("OS", _clean_os(info.get("os", {}))),
        ("Python", _clean_python(info.get("python", {}))),
        ("CPU", _clean_cpu(info.get("cpu", {}))),
        ("RAM", _fmt_gb(info.get("memory", {}).get("total_gb"))),
    ]
    rows.extend(_clean_gpu_rows(info.get("gpu", {})))
    return pd.DataFrame(rows, columns=["component", "value"]).dropna()


def _package_table(info: dict[str, Any]) -> pd.DataFrame:
    packages = info.get("packages", {})
    rows = [
        {"package": name, "version": version}
        for name, version in packages.items()
        if version
    ]
    return pd.DataFrame(rows)


def _clean_os(os_info: dict[str, Any]) -> str:
    system = os_info.get("system")
    release = os_info.get("release")
    machine = os_info.get("machine")
    if system and release:
        return f"{system} {release} ({machine})" if machine else f"{system} {release}"
    return os_info.get("platform", "")


def _clean_python(py_info: dict[str, Any]) -> str:
    impl = py_info.get("implementation", "Python")
    version = py_info.get("version", "")
    executable = py_info.get("executable_name")
    suffix = f" via {executable}" if executable else ""
    return f"{impl} {version}{suffix}".strip()


def _clean_cpu(cpu: dict[str, Any]) -> str:
    brand = cpu.get("brand") or _fallback_cpu_brand(cpu.get("processor"))
    cores = []
    if cpu.get("physical_cores") is not None:
        cores.append(f"{cpu['physical_cores']} physical")
    if cpu.get("logical_cores") is not None:
        cores.append(f"{cpu['logical_cores']} logical")
    core_text = f" ({', '.join(cores)} cores)" if cores else ""
    return f"{brand}{core_text}" if brand else core_text.strip()


def _fallback_cpu_brand(processor: str | None) -> str | None:
    if not processor:
        return None
    match = re.search(r"Family\s+(\d+)\s+Model\s+(\d+)", processor)
    if "GenuineIntel" in processor and match:
        return f"Intel CPU family {match.group(1)} model {match.group(2)}"
    return processor


def _clean_gpu_rows(gpu: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    smi_devices = gpu.get("nvidia_smi", [])
    torch_info = gpu.get("torch", {})
    torch_devices = torch_info.get("devices", [])
    devices = smi_devices or torch_devices
    for idx, dev in enumerate(devices):
        name = dev.get("display_name") or _gpu_display_name(dev.get("name", "GPU"))
        details = []
        if dev.get("architecture"):
            details.append(dev["architecture"])
        if dev.get("compute_capability"):
            details.append(f"compute {dev['compute_capability']}")
        memory = dev.get("memory_total_gb")
        if memory is None and dev.get("memory_total_mb") is not None:
            memory = round(float(dev["memory_total_mb"]) / 1024, 2)
        if memory is not None:
            details.append(f"{memory:.1f} GB VRAM")
        if dev.get("driver_version"):
            details.append(f"driver {dev['driver_version']}")
        rows.append((f"GPU {idx}", f"{name} ({'; '.join(details)})"))

    if torch_info:
        cuda = torch_info.get("compiled_cuda")
        available = torch_info.get("cuda_available")
        version = torch_info.get("version")
        rows.append(("PyTorch/CUDA", f"torch {version}, CUDA {cuda}, available={available}"))
        arch = torch_info.get("arch_list") or []
        if arch:
            rows.append(("CUDA kernels", ", ".join(arch)))
    return rows


def _gpu_display_name(name: str) -> str:
    match = re.search(r"\bRTX\s+\d{3,5}(?:\s+(?:Ti|SUPER))?\b", name, flags=re.I)
    if match:
        return match.group(0).upper().replace("  ", " ")
    cleaned = re.sub(r"^NVIDIA\s+", "", name, flags=re.I)
    cleaned = re.sub(r"^GEFORCE\s+", "", cleaned, flags=re.I)
    return cleaned.strip() or name


def _fmt_gb(value: Any) -> str | None:
    if value is None:
        return None
    return f"{float(value):.1f} GB"


def _benchmark_metadata(spec: BenchmarkSpec, cfg: dict[str, Any]) -> pd.DataFrame:
    data = cfg.get("data", {})
    rows = [
        ("config", _relpath(spec.config_path)),
        ("results", _relpath(spec.result_dir)),
        ("data source", data.get("source")),
        ("sample", _sample_text(data, cfg)),
        ("horizon", cfg.get("horizon")),
        ("latency enabled", bool(cfg.get("latency", False))),
    ]
    return pd.DataFrame(rows, columns=["field", "value"])


def _sample_text(data: dict[str, Any], cfg: dict[str, Any]) -> str:
    if "oos_year" in cfg:
        start = data.get("start")
        end = data.get("end")
        return f"{start} to {end}; OOS {cfg['oos_year']}"
    return f"{data.get('start')} to {data.get('end')}; OOS {cfg.get('oos_start')} to {cfg.get('oos_end')}"


def _model_coverage_table(cfg: dict[str, Any], summary: pd.DataFrame) -> pd.DataFrame:
    configured = configured_result_models(cfg)
    actual = set(summary["model"].astype(str)) if not summary.empty and "model" in summary else set()
    rows = []
    for family in ["Chronos-Bolt", "Chronos-2", "TimesFM", "TimesFM XReg"]:
        models = [model for model in configured if _model_family(model) == family]
        rows.append(
            {
                "family": family,
                "configured models": ", ".join(models) if models else "-",
                "present in current artifacts": ", ".join([m for m in models if m in actual]) if models else "-",
            }
        )
    return pd.DataFrame(rows)


def _summary_view(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "family",
        "r2_oos",
        "mae",
        "rmse",
        "dir_acc",
        "f1",
        "sharpe",
        "ann_return",
        "max_dd",
        "n_obs",
        "n_dates",
    ]
    out = summary.copy()
    out["family"] = out["model"].map(_model_family)
    out = out[[c for c in cols if c in out.columns]]
    return _round_for_display(out)


def _latency_view(latency: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "family",
        "fit_seconds",
        "predict_mean",
        "p50",
        "p95",
        "p99",
        "assets_per_sec",
        "bars_per_sec",
        "n_calls",
    ]
    out = latency.copy()
    out["family"] = out["model"].map(_model_family)
    out = out[[c for c in cols if c in out.columns]]
    return _round_for_display(out)


def _round_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        if col in {"n_obs", "n_dates", "n_calls"}:
            out[col] = out[col].astype("Int64")
        else:
            out[col] = out[col].round(4)
    return out


def _load_csv_with_model(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "model" not in df.columns and df.columns.size:
        df = df.rename(columns={df.columns[0]: "model"})
    return df


def _relpath(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _result_model_name(name: str) -> str:
    return re.sub(r"-(?:cpu|cuda|auto)$", "", name)


def _model_family(model: str) -> str:
    if model in {"zero", "mean"}:
        return "Simple baseline"
    if model in {"ridge", "lightgbm", "catboost"}:
        return "Supervised baseline"
    if model.startswith("chronos-bolt"):
        return "Chronos-Bolt"
    if model.startswith("chronos-2"):
        return "Chronos-2"
    if model.startswith("timesfm-2.5-xreg"):
        return "TimesFM XReg"
    if model.startswith("timesfm-2.5"):
        return "TimesFM"
    if model.startswith("moirai"):
        return "MOIRAI"
    return "Other"


def _display_images(spec: BenchmarkSpec, image_names: list[str]) -> None:
    display, Markdown = _display_tools()
    Image = _image_tool()
    shown = False
    for image_name in image_names:
        image_path = spec.result_dir / image_name
        if image_path.exists():
            shown = True
            display(Markdown(f"**{image_name}**"))
            display(Image(filename=str(image_path)))
    if not shown:
        display(Markdown("No plot files found for this benchmark yet."))


def _display_tools():
    from IPython.display import Markdown, display

    return display, Markdown


def _image_tool():
    from IPython.display import Image

    return Image
