"""Reproduce the final project artifacts.

Stages are intentionally explicit so the pipeline fails loudly when required
WRDS credentials or generated data are missing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PULL_MODULES = [
    "src.pulls.pull_crsp",
    "src.pulls.pull_compustat",
    "src.pulls.pull_linking",
    "src.pulls.pull_riskfree",
    "src.pulls.pull_factors",
    "src.pulls.pull_macro",
]

FEATURE_MODULES = [
    "src.features.build_targets",
    "src.features.build_returns",
    "src.features.build_momentum",
    "src.features.build_volatility",
    "src.features.build_liquidity",
    "src.features.build_fundamentals",
    "src.features.build_macro_features",
    "src.features.build_cross_sectional",
]

FINAL_CONFIGS = [
    "configs/wrds_daily_2022_full.yaml",
    "configs/wrds_daily_2023_full.yaml",
    "configs/ff25_2023.yaml",
    "configs/intraday_1m_wrds.yaml",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce FINM-33200 final project artifacts")
    p.add_argument(
        "--stage",
        choices=["all", "auth", "pull", "features", "panel", "bench", "report"],
        default="all",
    )
    p.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Benchmark config to run for --stage bench. May be repeated.",
    )
    p.add_argument("--no-plots", action="store_true", help="Forwarded to bench.run")
    p.add_argument(
        "--wait-auth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Poll WRDS auth while you approve Duo/MFA. Enabled by default.",
    )
    p.add_argument(
        "--auth-timeout",
        type=int,
        default=300,
        help="Seconds to wait for WRDS auth/MFA before failing.",
    )
    p.add_argument(
        "--auth-interval",
        type=int,
        default=15,
        help="Seconds between WRDS auth polling attempts.",
    )
    args = p.parse_args()

    _load_dotenv_if_available()
    stages = ["auth", "pull", "features", "panel", "bench", "report"] if args.stage == "all" else [args.stage]
    auth_checked = False
    for stage in stages:
        if stage == "auth":
            _require_wrds_env()
            _check_wrds_auth(
                wait=args.wait_auth,
                timeout_seconds=args.auth_timeout,
                interval_seconds=args.auth_interval,
            )
            auth_checked = True
        elif stage == "pull":
            _require_wrds_env()
            if not auth_checked:
                _check_wrds_auth(
                    wait=args.wait_auth,
                    timeout_seconds=args.auth_timeout,
                    interval_seconds=args.auth_interval,
                )
                auth_checked = True
            for module in PULL_MODULES:
                _run_module(module)
        elif stage == "features":
            for module in FEATURE_MODULES:
                _run_module(module)
        elif stage == "panel":
            _run_module("src.datasets.build_panel_dataset")
        elif stage == "bench":
            configs = args.configs or FINAL_CONFIGS
            for cfg in configs:
                cmd = [sys.executable, "-m", "bench.run", "--config", cfg]
                if args.no_plots:
                    cmd.append("--no-plots")
                _run(cmd)
        elif stage == "report":
            _run_report()


def _run_module(module: str) -> None:
    _run([sys.executable, "-m", module])


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _run_report() -> None:
    (ROOT / "docs").mkdir(exist_ok=True)
    _run([
        sys.executable,
        "-m",
        "bench.system_info",
        "--output",
        "results/system_info.json",
    ])
    notebook = "notebooks/report.ipynb"
    _run([
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--inplace",
        notebook,
    ])
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "html",
        "--output-dir",
        "docs",
        "--output",
        "index",
        notebook,
    ]
    _run(cmd)


def _require_wrds_env() -> None:
    missing = [name for name in ("WRDS_USERNAME", "WRDS_PASSWORD") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "WRDS credentials are required for the canonical pipeline. "
            f"Missing: {', '.join(missing)}. Add them to .env or the shell environment."
        )


def _check_wrds_auth(
    *,
    wait: bool,
    timeout_seconds: int,
    interval_seconds: int,
) -> None:
    from src.utils.wrds_connection import get_connection

    deadline = time.monotonic() + max(timeout_seconds, 1)
    attempt = 1

    if wait:
        print(
            "Waiting for WRDS auth. Approve the Duo push if it appears; "
            f"retrying for up to {timeout_seconds}s.",
            flush=True,
        )

    while True:
        conn = get_connection()
        try:
            df = conn.query("SELECT 1 AS ok")
            print(df.to_string(index=False), flush=True)
            return
        except Exception as exc:
            if not wait or time.monotonic() >= deadline:
                raise
            remaining = int(deadline - time.monotonic())
            print(
                f"WRDS auth not accepted yet; retrying in {interval_seconds}s "
                f"({remaining}s left).",
                flush=True,
            )
            time.sleep(max(interval_seconds, 1))
            attempt += 1
        finally:
            conn.close()


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


if __name__ == "__main__":
    main()
