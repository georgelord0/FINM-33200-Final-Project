# Data Pipeline & Feature Documentation

Data pipeline for benchmarking multivariate Time Series Foundation Models (TSFMs) on financial return prediction.

## Setup

### 1. Install dependencies

```bash
cd src
pip install -e .
# or
pip install -r requirements.txt
```

### 2. Configure WRDS credentials

Copy the example env file and fill in your WRDS credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password
DATA_DIR=./data
```

Alternatively, set up WRDS via `pgpass` (one-time):

```bash
python -c "import wrds; wrds.Connection()"
```

---

## Pipeline Overview

```
WRDS Pulls                    Feature Builders              Dataset Assembly
----------                    ----------------              ----------------
pull_crsp ──────────────┬──── build_returns ──────────┐
                        ├──── build_momentum ─────────┤
                        ├──── build_volatility ───────┤
                        ├──── build_liquidity ────────┤
                        ├──── build_cross_sectional ──┤
                        └──── build_targets ──────────┤
                                                      ├──── build_panel
pull_riskfree ────────────── build_targets ───────────┤
                                                      │
pull_factors ─────────────┬── build_volatility ───────┤
                          └── build_cross_sectional ──┤
                                                      │
pull_compustat ───┐                                   │
pull_linking ─────┴────────── build_fundamentals ─────┤
                                                      │
pull_macro ───────────────── build_macro_features ────┘
```

---

## Stage 1: Data Pulls

Each pull script connects to WRDS, downloads raw data, normalizes it in-place (type casting, deduplication, null handling, schema validation), and writes research-ready parquet files. CRSP and Compustat pulls are chunked by year and cached — re-running skips years already on disk.

```bash
# Core data (required)
python -m src.pulls.pull_crsp        # CRSP daily stock data -> data/crsp/
python -m src.pulls.pull_compustat   # Compustat fundamentals -> data/compustat/
python -m src.pulls.pull_linking     # CRSP-Compustat link table -> data/linking/
python -m src.pulls.pull_riskfree    # Risk-free rate + FF3 -> data/riskfree/
python -m src.pulls.pull_factors     # Fama-French + momentum factors -> data/factors/

# Optional (gracefully skip if you lack access)
python -m src.pulls.pull_macro       # FRED macro series -> data/macro/
python -m src.pulls.pull_options     # OptionMetrics IV/greeks -> data/options/
python -m src.pulls.pull_ibes        # I/B/E/S analyst estimates -> data/ibes/
```

### Data Sources

| Source | WRDS Table | Records | Frequency | Coverage |
|--------|-----------|---------|-----------|----------|
| CRSP | `crsp.dsf` + `crsp.dsenames` | 43.7M | Daily | 1990-01-02 to 2024-12-31 |
| Compustat | `comp.funda` | 331K | Annual | 1990-01-31 to 2026-03-31 |
| Fama-French Factors | `ff.factors_daily` | 8.8K | Daily | 1990-01-02 to 2024-12-31 |
| Risk-Free Rate | `ff.factors_daily` | 9.1K | Daily | 1990-01-02 to 2026-03-31 |
| Macro (FRED) | `pandas_datareader` FRED API | 9.3K | Daily (ffilled) | 1990-01-01 to 2024-12-31 |
| CCM Linking | `crsp.ccmxpf_linktable` | 40.5K | Static | All available |
| I/B/E/S Estimates | `ibes.statsum_epsus` | 114K | Periodic | 1990 - 2024 |
| I/B/E/S Recommendations | `ibes.recddet` | 165K | Periodic | 1990 - 2024 |
| OptionMetrics | `optionm.opprcd` | 0 | Daily | Not accessible |

### Universe Filters (CRSP)

- Common stocks only (share codes 10, 11)
- Major exchanges (NYSE, AMEX, NASDAQ)
- Non-zero, non-null prices
- Returns capped at |ret| <= 3.0

---

## Stage 2: Feature Engineering

Feature builders are pure functions that read normalized parquet files, compute features with no look-ahead bias, and write to `data/features/`.

```bash
python -m src.features.build_targets         # excess returns, direction labels
python -m src.features.build_returns          # lagged & cumulative returns
python -m src.features.build_momentum         # 1m/3m/6m/12m momentum, reversal
python -m src.features.build_volatility       # realized vol, downside vol, beta
python -m src.features.build_liquidity        # turnover, Amihud, volume z-score
python -m src.features.build_fundamentals     # B/M, ROE, ROA, leverage, investment
python -m src.features.build_macro_features   # yield spread, inflation, regimes
python -m src.features.build_cross_sectional  # relative returns, ranks, z-scores
```

### Targets (`targets.parquet`)

Built from **CRSP** daily returns + **FF risk-free rate**.

| Feature | Formula | Lag |
|---------|---------|-----|
| `excess_ret` | ret - rf | 0 (same-day) |
| `target_ret` | next-day excess return | shift(-1) forward-looking |
| `direction` | 1 if target_ret > 0, else 0 | forward-looking |

### Return Features (`return_features.parquet`)

Built from **CRSP** daily returns and prices.

| Feature | Formula | Lag |
|---------|---------|-----|
| `ret_lag_1` | Return at t-1 | 1 day |
| `ret_lag_5` | Return at t-5 | 5 days |
| `ret_lag_21` | Return at t-21 | 21 days |
| `cum_ret_5` | Cumulative 5-day return | 1 day |
| `cum_ret_21` | Cumulative 21-day return | 1 day |
| `cum_ret_63` | Cumulative 63-day return | 1 day |
| `log_ret` | ln(P_t / P_{t-1}) | 0 |

### Momentum Features (`momentum_features.parquet`)

Built from **CRSP** daily returns.

| Feature | Formula | Lag |
|---------|---------|-----|
| `mom_1m` | 21-day cumulative return | 1 day |
| `mom_3m` | 63-day cumulative return | 1 day |
| `mom_6m` | 126-day cumulative return | 1 day |
| `mom_12m` | 252-day cumulative return | 1 day |
| `short_term_reversal` | 5-day cumulative return | 1 day |
| `mom_12_1` | 12-month return (lagged 21d) minus 1-month return (lagged 1d) | 1-21 days |

### Volatility Features (`volatility_features.parquet`)

Built from **CRSP** daily returns + **FF Factors** (market excess return).

| Feature | Formula | Source | Lag |
|---------|---------|--------|-----|
| `rvol_21` | 21-day rolling std of returns | CRSP | 1 day |
| `rvol_63` | 63-day rolling std of returns | CRSP | 1 day |
| `downside_vol_63` | 63-day std of negative returns only | CRSP | 1 day |
| `beta_252` | Rolling CAPM beta: Cov(r_i, r_m) / Var(r_m) over 252 days | CRSP + FF `mktrf` | 1 day |

### Liquidity Features (`liquidity_features.parquet`)

Built from **CRSP** daily data (price, volume, shares outstanding, returns).

| Feature | Formula | Lag |
|---------|---------|-----|
| `dollar_volume` | \|price\| * volume | 0 |
| `turnover` | volume / shares_outstanding | 1 day |
| `amihud_21` | 21-day rolling mean of \|return\| / dollar_volume | 1 day |
| `volume_zscore_63` | z-score of log(volume) over 63-day rolling window | 1 day |

### Fundamental Features (`fundamental_features.parquet`)

Built from **Compustat** annual fundamentals + **CRSP** market cap, linked via the **CCM linking table**. All fundamentals are lagged 90 calendar days from `datadate` to account for SEC filing delay.

| Feature | Formula | Source Fields | Lag |
|---------|---------|---------------|-----|
| `book_to_market` | ceq / market_cap (forward-filled daily) | Compustat `ceq` + CRSP `market_cap` | 90 days |
| `roe` | ni / ceq | Compustat `ni`, `ceq` | 90 days |
| `roa` | ni / at | Compustat `ni`, `at` | 90 days |
| `gross_profitability` | (sale - cogs) / at | Compustat `sale`, `cogs`, `at` | 90 days |
| `leverage` | (dltt + dlc) / at | Compustat `dltt`, `dlc`, `at` | 90 days |
| `asset_growth` | at / at_lag - 1 (year-over-year) | Compustat `at` | 90 days |

### Macro Features (`macro_features.parquet`)

Built from **FRED** macroeconomic series (via `pandas_datareader` fallback).

| Feature | Formula | FRED Series | Lag |
|---------|---------|-------------|-----|
| `term_spread` | GS10 - TB3MS | `GS10`, `TB3MS` | 1 day |
| `inflation_change` | pct_change(CPI) | `CPIAUCSL` | 1 day |
| `unemployment_change` | diff(UNRATE) | `UNRATE` | 1 day |
| `high_vix` | 1 if VIX > expanding median | `VIXCLS` | 1 day |
| `inverted_yield_curve` | 1 if (GS10 - TB3MS) < 0 | `GS10`, `TB3MS` | 1 day |
| `rising_unemployment` | 1 if 3-month change in UNRATE > 0 | `UNRATE` | 1 day |
| `high_inflation` | 1 if 12-month CPI change > expanding median | `CPIAUCSL` | 1 day |

### Cross-Sectional Features (`cross_sectional_features.parquet`)

Built from **CRSP** daily returns + **FF Factors** (market return).

| Feature | Formula | Lag |
|---------|---------|-----|
| `market_relative_ret` | ret - market_return | 0 |
| `ret_rank` | Cross-sectional percentile rank of returns (0-1) per date | 0 |
| `ret_zscore` | Cross-sectional z-score of returns per date | 0 |

---

## Stage 3: Dataset Construction

```bash
# Canonical panel: (date, permno, target, features)
python -m src.datasets.build_panel_dataset

# Multivariate tensors for TSFMs: (batch, context_window, n_features)
python -m src.datasets.build_multivariate_dataset \
    --panel-path data/datasets/panel.parquet \
    --context-window 252 \
    --train-end 2018-12-31 \
    --val-end 2020-12-31
```

## Stage 4: Model Training

Not yet implemented. The pipeline produces research-ready datasets for TSFM ingestion.

---

## Expected Outputs

```
data/
├── crsp/          # year-partitioned daily stock data
├── compustat/     # year-partitioned annual fundamentals
├── linking/       # ccm_link.parquet
├── riskfree/      # ff_factors_daily.parquet
├── factors/       # factors_daily.parquet
├── macro/         # macro_daily.parquet
├── options/       # year-partitioned IV/greeks (if available)
├── ibes/          # analyst estimates (if available)
├── features/      # computed feature parquets
└── datasets/
    ├── panel.parquet           # full panel dataset
    └── multivariate_*.npz     # tensor datasets
```

## Look-Ahead Bias Prevention

- All time-series features use `shift(1)` or greater lags
- Target variable uses `shift(-1)` (explicitly forward-looking)
- Fundamental features lagged 90 calendar days for SEC filing delay
- Panel builder includes a leakage check verifying `target_ret[t] == ret[t+1]`
- Forward-fill is limited to fundamentals with a 126-day max gap
- Train/val/test splits are strictly temporal (no shuffling)

## Configuration

| File | Purpose |
|------|---------|
| `configs/paths.yaml` | Data directory paths |
| `configs/features.yaml` | Feature engineering parameters (lags, windows) |
| `configs/assets.yaml` | Universe filters (share codes, exchanges, min price) |
| `.env` | WRDS credentials and data directory |

## Requirements

- Python 3.11+
- WRDS account with access to CRSP, Compustat, and Fama-French libraries
- Optional: OptionMetrics and I/B/E/S access for additional features
