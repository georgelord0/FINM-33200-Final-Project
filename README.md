# FINM-33200 Final Project

Benchmarking covariate-supported time-series foundation models on US equity
return forecasting. Extends Rahimikia, Ni and Wang (2025), "Re(Visiting) Time
Series Foundation Models in Finance" (arXiv:2511.18578), which only evaluated
univariate TSFMs and flagged multivariate models as future work.

We test multivariate / covariate-aware TSFMs (TimesFM 2.5, Chronos2, MOIRAI)
against linear and tree baselines on US daily panels (S&P 500 via yfinance
and Fama-French 25 portfolios). Plan also covers a short-horizon (1s to 1m)
extension for runtime / latency analysis.

## Team

- Max: data collection (returns + covariates)
- George: foundation model wrappers (standardised predict interface)
- Charles: benchmark harness, baselines, metrics, results (this slice)
- Cesare: short-horizon benchmark + latency

## Layout

```
bench/                 Harness package
  protocols.py           Panel + Forecaster types
  metrics.py             R^2_oos, MAE, RMSE, dir-acc, F1, L/S portfolio stats
  walkforward.py         Expanding-window driver (run, run_window)
  baselines.py           Zero, Mean, Ridge, LightGBM, CatBoost
  chronos_bolt.py        Chronos-Bolt univariate zero-shot wrapper
  chronos_two.py         Chronos-2 multivariate zero-shot wrapper
  latency.py             TimedForecaster, percentile summary, plots
  data_yfinance.py       S&P 500 daily data via yfinance + Ken French volume
  data_ff25.py           Fama-French 25 portfolios (Ken French)
  data_intraday.py       Intraday dispatcher (source → loader)
  data_yf_intraday.py    yfinance 1m loader (smoke only; ~30d history limit)
  data_wrds.py           WRDS TAQ tick→bar loader (.env: WRDS_USERNAME/PASSWORD)
  data_synthetic.py      Synthetic intraday panel (latency benchmarks, tests)
  covariates_intraday.py Backward-only bar-count covariate builder
  report.py              CSV + markdown summary
  plots.py               Sharpe-by-model bar, L/S equity curves
  aggregate.py           Cross-run summary combiner
  run.py                 CLI: python -m bench.run --config ...
configs/               YAML run configs (daily, intraday 1m/1s, latency)
tests/                 pytest suite (synthetic fixture, leakage guards,
                       intraday, latency, offline WRDS)
```

## Reproduce

```sh
pip install -r requirements.txt
pytest                                                # 24 tests

# Daily replication (Charles's slice)
python -m bench.run --config configs/smoke.yaml       # 10-ticker exploratory
python -m bench.run --config configs/replication_us_2022.yaml
python -m bench.run --config configs/ff25_2023.yaml   # FF25 daily portfolios
```

### Short-horizon + latency (Cesare's slice)

```sh
# Synthetic — no network or credentials, deterministic timing
OMP_NUM_THREADS=1 python -m bench.run --config configs/latency_synthetic.yaml

# yfinance 1m smoke — recent ~7-day window only (yfinance limit)
OMP_NUM_THREADS=1 python -m bench.run --config configs/intraday_1m_smoke.yaml

# WRDS TAQ — institutional millisecond ticks resampled to 1s / 1m bars.
# Needs WRDS_USERNAME and WRDS_PASSWORD in .env (see .env.example).
OMP_NUM_THREADS=1 python -m bench.run --config configs/intraday_1s_wrds.yaml
OMP_NUM_THREADS=1 python -m bench.run --config configs/intraday_1m_wrds.yaml
```

**macOS reminder**: PyTorch and LightGBM both ship libomp; loading both in
one process segfaults on macOS. `OMP_NUM_THREADS=1` serializes OpenMP and
avoids the race. Linux / CUDA boxes don't need it.

**CUDA models**: Chronos wrappers default to `device="cuda"`. On a
CPU-only box, append `-cpu` to the model name in the config (e.g.
`chronos-bolt-60-cpu`); the `chronos-{family}-{ctx}[-cpu|-cuda]` regex in
[bench/run.py](bench/run.py) handles dispatch.

Outputs land in `results/<config-name>/`:
- `forecasts.csv`: long-format (date, asset_id, model, forecast, realized)
- `summary.csv` / `summary.md`: per-model accuracy + L/S portfolio table
- `sharpe_by_model.png`, `equity_curves.png`
- when `latency: true`: `latency.csv` / `latency.md`, `latency_p95.png`,
  `latency_throughput.png` (per-call mean / p50 / p95 / p99, plus
  `assets_per_sec` and `bars_per_sec` throughput)

### Data sources

| Source | Granularity | Cost | Notes |
|---|---|---|---|
| `yfinance` | daily | free | survivorship-biased SP500 |
| `ff25` | daily | free | Fama-French 25 portfolios |
| `yfinance_intraday` | 1m | free | ~7d/request, ~30d total history |
| `wrds` | 1s, 1m | institutional | needs `WRDS_USERNAME` + `WRDS_PASSWORD`; TAQ `taqm_<year>` |
| `synthetic_intraday` | configurable | free | AR(1)+factor; for latency / tests |

## Metric definitions

- `r2_oos`: `1 - sum((r - r_hat)^2) / sum(r^2)` pooled over all (date, asset),
  matching Eq. 13 of the paper (Gu et al. 2020 convention). Benchmarks against
  a zero prediction, not the historical mean.
- `mae`, `rmse`: standard absolute / squared error metrics.
- `dir_acc`: share of correct sign predictions.
- `f1`: macro-F1 over {up, down}.
- L/S portfolio: daily decile sort by forecast; long top decile minus short
  bottom decile, equal-weighted, no transaction costs. Returns NaN when
  forecasts have no within-date dispersion (e.g. Zero baseline).
- `ann_return`, `ann_vol`, `sharpe`: mean / std × `bars_per_year` and
  `sqrt(bars_per_year)`. Set per config: 252 (daily, default), 98 280 = 390 × 252
  (1-minute), 5 896 800 = 23 400 × 252 (1-second).
- `max_dd`: minimum of cumulative-return drawdown series.

## Notes

- yfinance is the exploratory data source. Survivorship bias from current
  S&P 500 membership is acknowledged; replace via the same `Panel` interface
  once a clean CRSP pipeline is available.
- TSFM wrappers plug in as additional `Forecaster` implementations. The
  protocol in `bench/protocols.py` is provisional and may add a
  quantile-return mode once integration starts.
- `_aligned_xy` in `bench/baselines.py` filters training rows by target date
  (not feature date) to keep OOS-year returns out of training labels. See
  `tests/test_leakage.py` for the regression test.
