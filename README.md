# Aegis

A risk engine that puts **calibrated uncertainty** around market-risk forecasts. A semantic parser (deterministic lexicon or a local SLM) turns financial text into structured shock signals; a factor engine forecasts portfolio volatility and Value-at-Risk; a conformal prediction layer wraps those forecasts in intervals whose coverage is checked, not assumed.

The headline result is an honest one. Textbook split conformal **misses its 95% target** on 2020–2026 market data, because markets aren't exchangeable. Adaptive Conformal Inference recovers 95%, including through the COVID crash. Text signals did **not** improve real-data forecasts; the ablation shows exactly when and why.

See [docs/PLAN.md](docs/PLAN.md) for the original design, the milestone log and which success criteria were met.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e ".[dev,dashboard]"
```

```bash
python main.py demo
```

`demo` runs the full pipeline offline on a synthetic market in about 3 seconds and writes `reports/demo/backtest_report.md`. Then:

```bash
python main.py dashboard
```

## Commands

| Command | What it does |
|---|---|
| `demo` | Offline end-to-end run on a synthetic world (no network, no model) |
| `fetch -c configs/default.yaml` | Download and cache prices (yfinance) and FOMC statements |
| `backtest -c <config>` | Walk-forward backtest; writes series, `metrics.json`, figures and a markdown report to `reports/<name>/` |
| `report -c <config>` | Re-render the report from saved series |
| `ablation -c <config>` | Backtest with and without text signals (plus SNR sweep and oracle on synthetic data) |
| `parse -c <config>` | Run the parser over the news store and print signal statistics |
| `bench` | Latency benchmark; writes `reports/benchmark.json` |
| `dashboard` | Streamlit app over every saved run |
| `config -c <config>` | Print the resolved configuration |

Any config value can be overridden with `--set key.path=value`, e.g. `--set conformal.alpha=0.1`.

Configs: `default.yaml` (real data, price-only), `synthetic.yaml` (known-signal world with news), `ollama.yaml` (real data + local SLM).

## How it works

```
 prices (yfinance) ─┐                    ┌─ FOMC statements / synthetic headlines
                    │                    │   masked → parsed → RiskSignal (validated JSON)
                    │                    │   → daily as-of aggregation → clipped to train range
                    ▼                    ▼
       Factor engine: EWMA covariance × semantic shock overlay
       HAR / ridge / LightGBM point model μ̂, scale σ̂, quantiles q̂_lo, q̂_hi
                    │
                    ▼
       Conformal layer: split · normalized · CQR · rolling · ACI · one-sided VaR
                    │
                    ▼
       Walk-forward backtest → coverage, Winkler, Kupiec, Christoffersen → report / dashboard
```

**Targets.** (A) log realized portfolio volatility over the next 5 days, with two-sided 95% intervals; (B) next-day portfolio return, with 95% and 99% one-sided VaR.

**Conformal.** Calibration scores S = |y − μ̂| (or normalized / CQR variants) give the threshold q̂ = the ⌈(n+1)(1−α)⌉-th smallest score. Under exchangeability this guarantees 1 − α ≤ coverage ≤ 1 − α + 1/(n+1), marginally. ACI updates its working level α_t ← α_t + γ(α − miss_t) after every outcome, which guarantees long-run coverage for *any* sequence, with no exchangeability assumption.

**No look-ahead.** A news item counts toward day t only if it was published before 4 p.m. New York time on day t. Splits have a 5-day embargo, and online methods see an outcome only after it is realized. A test perturbs every price after a test date t and asserts that every interval and VaR dated on or before t is unchanged.

**Guarantees and limits.** Split conformal's guarantee does not hold on market data, and the report says so. ACI's guarantee is long-run, not per quarter. Coverage by volatility regime (conditional coverage) is reported but not guaranteed by any method here.

## Results (real data, price-only)

Train 2010–2017, calibrate 2018–2019, test 2020-01 → 2026-10 (1,712 days), equal-weight SPY/TLT/GLD/USO, point model: standardized ridge with a CV-chosen penalty.

| 5-day vol interval, target 95% | Coverage | 2020 | Worst quarter | Avg width | Winkler |
|---|---|---|---|---|---|
| Gaussian (uncalibrated) | 94.0% | 89.7% | 71.4% | 1.648 | 2.149 |
| Split conformal | 93.3% | 88.9% | 68.3% | 1.596 | 2.162 |
| CQR | 94.3% | 87.4% | 65.1% | 1.802 | 2.301 |
| Rolling conformal | 94.8% | 91.7% | 73.0% | 1.722 | 2.184 |
| **Adaptive conformal (ACI)** | **95.0%** | **94.5%** | **82.5%** | 1.739 | 2.212 |

| One-day VaR | 95%: breaches · Kupiec · Christoffersen | 99%: breaches · Kupiec · Christoffersen |
|---|---|---|
| Split conformal | 5.49% · pass · fail | 0.82% · pass · pass |
| Rolling conformal | 5.02% · pass · fail | 0.99% · pass · pass |
| ACI | 4.91% · pass · **pass** | 0.99% · pass · fail |
| Gaussian EWMA (RiskMetrics) | 4.96% · pass · fail | 1.93% · **fail** · pass |
| Historical simulation | 5.78% · pass · fail | 1.69% · **fail** · fail |
| GARCH(1,1)-t | 5.72% · pass · fail | 1.40% · pass · fail |

Conformal VaR fixes the 99% tail that Gaussian and historical simulation underestimate. No method passes every test at both levels: rolling conformal is cleanest at 99%, and ACI is the only method without clustered breaches at 95%.

**What didn't work.** Volatility-scaled (normalized) and CQR intervals never beat the plain absolute residual: the target is already log-vol, whose residuals are close to homoskedastic, so a scale model mostly adds noise. For the same reason the online methods use the absolute score by default (ACI on CQR: Winkler 2.320 → on absolute: 2.212).

## Does the text layer help?

| Synthetic world (known signal) | Vol RMSE vs prices only | Parsed vs true shock (corr.) |
|---|---|---|
| Headlines are pure noise (SNR 0) | +1.9% | −0.02 |
| SNR 0.5 (default) | +0.9% | 0.15 |
| SNR 2 | +0.4% | 0.51 |
| Near noise-free headlines (SNR 100) | **−1.1%** | 0.90 |
| True latent shocks (oracle upper bound) | −2.2% | 1.00 |

| Real data, 139 FOMC statements | Vol RMSE vs prices only | CQR coverage · width | ACI coverage · width |
|---|---|---|---|
| Prices only | — | 94.3% · 1.80 | 95.0% · 1.74 |
| + lexicon text | +1.2% | 94.4% · 1.95 | 95.3% · 1.78 |
| + SLM text (qwen2.5:3b, prompt v2) | +2.8% | 94.4% · 2.50 | 95.1% · 1.79 |
| + SLM text, prompt v1, no clipping (before fixes) | +16.7% | 82.4% · 2.07 | — |

Text helps only when it is both frequent and nearly noise-free. About eight FOMC statements a year is too sparse, so `default.yaml` runs price-only. The lexicon's rules were written with knowledge of the synthetic headline templates, so on synthetic data the gap to the oracle is headline noise, not parsing error.

### Local SLM (Ollama)

Model output is constrained to the `RiskSignal` JSON schema, validated, retried once, and falls back to the lexicon on failure. There were 0 retries and 0 fallbacks across all runs. Results are cached in `data/slm_cache.sqlite`, keyed by backend, model, prompt version and text. Dates, years, named events and Fed chairs are masked before parsing, so the model can't recall what happened next.

**Prompt v1 → v2.** v1 used neutral factor names with a stated sign convention. qwen2.5:3b reversed it wherever the convention was counter-intuitive: "credit spreads tighten" was scored as rising credit stress every time. v2 names each field by its positive direction (`rate_pressure`, `credit_stress`, `geopolitical_tension`, …), describes what +1 and −1 mean, and uses balanced few-shot examples. On 200 synthetic headlines scored against ground truth:

| | SLM v1 | SLM v2 | Lexicon |
|---|---|---|---|
| Sign agreement on clear headlines | 69–72% | **100%** | 100% |
| Correlation with true shock, SNR 2 | 0.11 | **0.51** | 0.43 |
| Correlation with true shock, SNR 100 | 0.38 | **0.95** | 0.91 |

The SLM still misreads some easing in the zero-rate era as tightening. The 15 March 2020 emergency cut and the September 2012 QE3 launch both score rate pressure +0.7 to +0.8; 5 of 7 checked decisions are scored correctly.

To run it yourself:

```bash
brew install ollama
```

```bash
ollama serve &
```

```bash
ollama pull qwen2.5:3b
```

```bash
python main.py ablation -c configs/ollama.yaml
```

## Latency (Apple M4 Pro, `python main.py bench`)

| Measure | Value |
|---|---|
| Full backtest, 1,301 test days, all methods and baselines | 1.0 s |
| Score one day (point model + conformal interval), p50 / p95 | 0.39 / 0.56 ms |
| Lexicon parse per headline, p50 | 0.09 ms |
| SLM parse per headline (qwen2.5:3b), p50 / p95 | 1.8 / 1.9 s |

## Repository layout

```
main.py                    CLI entry (same as the `aegis` command)
configs/                   default · synthetic · ollama
src/aegis/
  config.py                typed, validated YAML config with --set overrides
  data/                    prices (yfinance + cache) · news store (as-of) · FOMC loader · synthetic world
  slm_parser.py            RiskSignal schema · masking · lexicon + Ollama backends · SQLite cache
  features.py              HAR/EWMA/VIX features · targets · as-of signal aggregation · clipping
  risk_engine.py           EWMA covariance · shock overlay · point/scale/quantile models
  conformal_ncm.py         non-conformity scores · split / rolling / ACI · walk-forward runner
  baselines.py             Gaussian, historical-simulation and GARCH VaR
  metrics.py               coverage · Winkler · Kupiec · Christoffersen
  backtest.py              splits with embargo · walk-forward run for both targets
  report.py                metrics.json · figures · markdown report
  ablation.py · bench.py   text ablation · latency benchmark
dashboard/app.py           Streamlit dashboard over reports/*
tests/                     133 offline tests + network and live-SLM tests
```

## Development

```bash
pytest
```

```bash
pytest -m "network or llm"
```

```bash
ruff check . && ruff format --check .
```

The default `pytest` run is offline. The `network` and `llm` markers add tests against yfinance, federalreserve.gov and a live Ollama model.
