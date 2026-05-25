# Results Directory

Canonical benchmark runs write artifacts here:

- `forecasts.csv`
- `summary.csv`
- `summary.md`
- `sharpe_by_model.png`
- `equity_curves.png`
- latency reports and plots for latency-enabled configs

The WRDS-backed results require WRDS credentials and are generated with:

```sh
python scripts/reproduce.py --stage bench
```

Raw WRDS source data and local caches are intentionally not committed.
