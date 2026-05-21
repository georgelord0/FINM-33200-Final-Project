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
  walkforward.py         Expanding-window driver
  baselines.py           Zero, Mean, Ridge, LightGBM, CatBoost
  chronos_bolt.py        Chronos-Bolt univariate zero-shot wrapper
  chronos_two.py         Chronos-2 multivariate zero-shot wrapper
  data_yfinance.py       S&P 500 daily data via yfinance + Ken French volume
  data_ff25.py           Fama-French 25 portfolios (Ken French)
  report.py              CSV + markdown summary
  plots.py               Sharpe-by-model bar, L/S equity curves
  aggregate.py           Cross-run summary combiner
  run.py                 CLI: python -m bench.run --config ...
configs/               YAML run configs
tests/                 pytest suite (synthetic fixture, leakage guards)
```

## Reproduce

```sh
pip install -r requirements.txt
pytest                                                # 14 tests
python -m bench.run --config configs/smoke.yaml       # 10-ticker exploratory run
python -m bench.run --config configs/replication_us_2022.yaml
python -m bench.run --config configs/ff25_2023.yaml   # FF25 daily portfolios
```

Outputs land in `results/<config-name>/`:
- `forecasts.csv`: long-format (date, asset_id, model, forecast, realized)
- `summary.csv` / `summary.md`: per-model metrics table
- `sharpe_by_model.png`, `equity_curves.png`

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
- `ann_return`, `ann_vol`, `sharpe`: daily mean / std times 252 / sqrt(252).
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
