# FINM-33200 Final Project

Benchmarking time-series foundation models for next-day US equity excess-return
forecasting.

The project asks whether zero-shot and covariate-aware TSFMs improve forecasts
relative to simple linear and tree baselines. The canonical pipeline is
WRDS-backed: `src/` pulls and builds the daily CRSP/Compustat/Fama-French panel,
and `bench/` evaluates models with a common walk-forward protocol. Fama-French
25 portfolios remain as a public-data sanity benchmark, and WRDS TAQ provides
the short-horizon latency extension.

Current default benchmark configs evaluate zero, mean, Ridge, LightGBM, CatBoost,
Chronos-Bolt, Chronos-2, TimesFM 2.5 target-only, and TimesFM 2.5 XReg. A MOIRAI
wrapper is implemented and registered, but it is not included in the default
full configs because the current Uni2TS dependency stack pins older Torch
versions that conflict with the RTX 50-series CUDA build used for the final run.

GitHub Pages report: build `docs/index.html` with the report stage, then publish
the repository from the `docs/` folder.

## Team

- Max: WRDS data collection and feature pipeline
- George: foundation model wrappers and standardized forecast interface
- Charles: benchmark harness, baselines, metrics, plots, result reporting
- Cesare: short-horizon benchmark and latency instrumentation

## Setup

Use Python 3.11 or newer. On the Windows/RTX 50-series environment used for the
current run, install from `requirements.txt`; it pins the CUDA 13 Torch wheel
needed for Blackwell GPUs.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For a lighter environment that only needs the package, report tooling, and
tests, use:

```powershell
python -m pip install -e ".[report,dev]"
```

If you are not using an NVIDIA RTX 50-series GPU, replace the Torch line in
`requirements.txt` with the platform-specific install command from PyTorch
before installing the model stack.

Verify CUDA when running the full TSFM benchmark:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
```

## WRDS Credentials

WRDS credentials are required for the canonical results. Create `.env` from
`.env.example`:

```text
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password
DATA_DIR=./src/data
HF_HUB_DISABLE_SYMLINKS_WARNING=true
```

Do not commit `.env`, raw WRDS data, local caches, or model weight caches.

### WRDS MFA

`scripts/reproduce.py` waits for Duo/MFA by default. Start the preflight, approve
the Duo push if one appears, and let the command continue polling:

```powershell
python scripts/reproduce.py --stage auth --auth-timeout 600 --auth-interval 10
```

If the preflight returns `ok = 1`, run the remaining stages. If it times out,
verify that `.env` contains your normal WRDS username and password.

## Reproduce

Run the full project pipeline:

```powershell
python scripts/reproduce.py --stage all
```

Run one stage at a time:

```powershell
python scripts/reproduce.py --stage pull
python scripts/reproduce.py --stage features
python scripts/reproduce.py --stage panel
python scripts/reproduce.py --stage bench
python scripts/reproduce.py --stage report
```

The full pipeline writes the licensed WRDS panel under `src/data/`, which is
ignored by Git and should not be committed. The report stage writes
`results/system_info.json`, executes `notebooks/report.ipynb`, and exports
`docs/index.html` with a cleaned runtime environment table covering OS, CPU,
GPU, RAM, CUDA, and key package versions.

Default benchmark configs run by `--stage bench`:

```powershell
python -m bench.run --config configs/wrds_daily_2022_full.yaml
python -m bench.run --config configs/wrds_daily_2023_full.yaml
python -m bench.run --config configs/ff25_2023.yaml
python -m bench.run --config configs/intraday_1m_wrds.yaml
```

Run a single config through the reproduction driver:

```powershell
python scripts/reproduce.py --stage bench --config configs/wrds_smoke.yaml --no-plots
```

CPU/local smoke checks:

```powershell
python -m bench.run --config configs/smoke.yaml --no-plots
python -m bench.run --config configs/wrds_smoke.yaml --no-plots
```

## Outputs

Benchmark artifacts land in `results/<config-name>/`:

- `forecasts.csv`: long-format forecasts and realized returns
- `summary.csv` and `summary.md`: accuracy and long-short portfolio metrics
- `sharpe_by_model.png` and `equity_curves.png`
- `latency.csv`, `latency.md`, and latency plots for latency-enabled configs;
  the WRDS intraday config records latency for baselines, Chronos, and TimesFM

Current generated result directories include:

- `results/wrds_daily_2022_full/`
- `results/wrds_daily_2023_full/`
- `results/ff25_2023/`
- `results/intraday_1m_wrds/`
- smoke/debug runs under `results/wrds_*_smoke/`

Raw WRDS source data and local caches are intentionally excluded from version
control.

## GitHub Pages

After running the report stage, commit the report output and publish Pages from
the `docs/` folder:

```powershell
git add README.md AI_USAGE.md requirements.txt pyproject.toml configs bench src scripts tests notebooks/report.ipynb docs/index.html results
git commit -m "Finalize benchmark report"
git push origin main
```

On GitHub, open the repository settings, choose **Pages**, set the source to
**Deploy from a branch**, select branch `main`, folder `/docs`, and save. Do not
add `.env`, `src/data/`, `.venv/`, `data_cache/`, or model caches.

## Project Layout

```text
src/
  pulls/                 WRDS pulls for CRSP, Compustat, linking, factors, macro
  features/              Leakage-controlled feature builders
  datasets/              Canonical long panel and tensor builders
bench/
  protocols.py           Shared Panel and Forecaster protocol
  data_wrds_panel.py     Adapter from src panel to bench Panel
  baselines.py           Zero, mean, Ridge, LightGBM, CatBoost
  chronos_bolt.py        Chronos-Bolt target-only wrapper
  chronos_two.py         Chronos-2 covariate-aware wrapper
  timesfm_wrappers.py    TimesFM 2.5 target-only and XReg wrappers
  moirai.py              Optional MOIRAI target-only wrapper via Uni2TS/GluonTS
  walkforward.py         Expanding-window OOS driver
  metrics.py             Accuracy and portfolio metrics
  report.py, plots.py    Result tables and plots
  report_notebook.py     Notebook display helpers for the generated report
  system_info.py         Runtime environment capture for the report
configs/                 Benchmark YAML configs
scripts/reproduce.py     End-to-end reproduction driver
notebooks/report.ipynb   Audience-facing report source
docs/index.html          GitHub Pages report output
tests/                   Unit and leakage tests
```

## Forecast Protocol

For each out-of-sample date, the model predicts next trading-day excess return
using information available after market close on the `asof` date. The realized
return is `excess_ret[t+1]`. The benchmark reports:

- `r2_oos`: pooled out-of-sample R2 against a zero-return forecast
- `mae`, `rmse`
- `dir_acc`, `f1`
- Long-short decile portfolio annualized return, volatility, Sharpe, and drawdown
- Latency percentiles and throughput for latency-enabled runs

## Data Sources

| Source | Use | Notes |
| --- | --- | --- |
| WRDS CRSP/Compustat/Fama-French | Canonical daily benchmark | Required for final reproduction |
| WRDS TAQ | Intraday 1m/1s latency extension | Required for short-horizon institutional run |
| Fama-French 25 portfolios | Public sanity benchmark | No WRDS credentials required |
| yfinance | Smoke/exploratory runs | Survivorship-biased current S&P 500 membership |
| Synthetic intraday | Tests and timing smoke | Not used for final empirical claims |

## Tests

```powershell
python -m pytest
```

The normal test suite uses small fixtures and fake model objects. It should not
download model weights or connect to WRDS. Full WRDS/model runs are manual,
slow integration checks driven by `scripts/reproduce.py`.

Useful focused checks:

```powershell
python -m pytest tests/test_model_registry.py tests/test_wrds_panel.py tests/test_timesfm_wrappers.py tests/test_chronos_moirai_wrappers.py
python -m bench.run --config configs/wrds_smoke.yaml --no-plots
```

## AI Usage

See [AI_USAGE.md](AI_USAGE.md).
