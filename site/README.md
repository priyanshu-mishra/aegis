# Aegis site

The public, guided site for Aegis. Plan: [docs/PLAN_v2.md](../docs/PLAN_v2.md), Parts B and C.

## Data contract (`public/data/*.json`, schema version 1)

Written by `python main.py export-site -c configs/default.yaml` from a saved backtest. The nightly
`Site` workflow (`.github/workflows/site.yml`) regenerates it, builds the site and deploys it to GitHub Pages; if the data download fails, the committed snapshot is used
and `status.json` says so.

Conventions:
- **Columnar time series:** `{"dates": ["YYYY-MM-DD", …], "series": {"name": [number | null, …]}}`.
- **Indexing:** dates are decision days. A row at date t was computed from information through the close of t, and its target lies after t.
- **Units:** log-vol series are log daily volatility (annualize with `exp(x) · √252`). Returns, VaR and ES levels are decimal returns; levels are negative in the tail.
- **Missing values:** non-finite numbers are `null`; an infinite interval is a pair of `null` bounds.

| File | Contents | Used by |
|---|---|---|
| `meta.json` | Run metadata: data range, splits, α, horizon, model, methods, strategy settings, git SHA, export time | Every chapter |
| `metrics.json` | All summary metrics (`report.compute_metrics`) | Scorecards, tables |
| `vol.json` | Test period: `y` (realized log vol), `mu` (point forecast), `ewma_log_vol`, `<method>_lo` / `_hi` | Intervals, coverage, 2020 replay |
| `calibration.json` | Calibration period: `y`, `mu`, `lo`, `hi`, `group` (vol tercile) | In-browser conformal playgrounds |
| `golden.json` | Python reference values (split q̂, ACI coverage and bounds) computed on the *rounded* exported data | JS correctness test |
| `aci_alpha.json` | ACI / DtACI working level α_t per test day | ACI chapter |
| `var.json` | Per tail level (`"0.05"`, `"0.01"`): `r_next`, each method's 1-day VaR level, and `<method>_breach` (1/0, computed at full precision) | VaR chapter |
| `es.json` | `r_next`, `<method>_var`, `<method>_es` at 97.5% | Regulatory view |
| `var_long.json` | `r_fwd_long` and each method's 10-day 99% VaR level | Regulatory view |
| `basel.json` | `<method>_exceptions`, `_multiplier`, `_capital` (rolling 250 days) | Traffic-light chart |
| `contrib.json` | Each asset's share of portfolio risk | Risk decomposition |
| `strategies.json` | `<strategy>_exp` (exposure) and `<strategy>_net` (next-day net return) | Portfolio chapter |
| `prices.json` | Weekly closes normalized to 100 | Meet the portfolio |
| `news.json` | FOMC statements: date, title, excerpt, lexicon-parsed shocks | Text chapter |
| `lexicon.json` | The lexicon parser's regex rules and exposures | Live headline parser |
| `ablation_<run>.json`, `sensitivity_<run>.json` | Text ablation and sensitivity grids | Text and appendix chapters |
| `benchmark.json`, `slm_accuracy*.json` | Latency and SLM accuracy (from local runs; CI keeps the committed copy) | Engineering appendix |
| `status.json` | Written by CI: `fresh` (bool), `checked` (UTC time), `data_last_day` | "Data as of" badge |
