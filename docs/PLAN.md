# Aegis — MVP Build Plan
### Conformal Risk Engine: SLM Semantic Parser → Quantitative Factor Engine → Conformal (NCM) Calibration

> Status: M0–M6 done (MVP complete; 133 offline + 2 network/LLM tests). Ollama 0.40.13 + qwen2.5:3b installed.

---

## 0. What this plan changes about the PRD (read first)

The PRD is a solid skeleton, but a few claims would get an MVP torn apart in review by anyone who knows conformal prediction or market risk. The plan fixes them up front and turns them into the project's **main story**.

| PRD claim | Problem | What we build instead |
|---|---|---|
| "Guaranteed 95% coverage", "exactly 1−α" | Split conformal gives **marginal** coverage `1−α ≤ P(y ∈ C) ≤ 1−α + 1/(n+1)` **only if the data are exchangeable**. Market returns are not: volatility clusters and regimes shift. | Implement the textbook split conformal method from the PRD **and** time-series-aware variants (rolling calibration, Adaptive Conformal Inference). Show, on real data, that the naive version under-covers in 2020/2022 and the adaptive one recovers ≈95%. That contrast is the headline result. |
| Symmetric `[μ̂ − q, μ̂ + q]` intervals | A constant width ignores volatility: too wide in calm markets, too narrow in crises. | Add the **normalized** non-conformity score `S = |y − μ̂| / σ̂` and **Conformalized Quantile Regression (CQR)**, so intervals widen when risk rises. |
| "Interval around point VaR" | VaR is already a quantile; putting an interval around it mixes two things. | Two clear targets: (1) **forecast of 5-day realized volatility** with a two-sided conformal interval; (2) **conformal VaR**: a one-sided lower bound on next-day portfolio return, backtested with standard VaR tests (Kupiec, Christoffersen). |
| "NCM = Next-Concept Modeling" | NCM in conformal literature means **Non-Conformity Measure**. | Use the standard term throughout. |
| SLM sentiment on historical news | (a) **Look-ahead leakage**: news timestamps must precede the forecast time. (b) **Memorization leakage**: Llama/Phi were trained on text describing what happened next. | Strict as-of timestamps, entity/date masking before parsing, and a fully **synthetic world** where we control the true signal, so the value of the text layer is measured honestly. |
| "No parsing failures" | Even with constrained decoding, models can time out or return out-of-range values. | Pydantic schema + constrained decoding + validation + retry + a deterministic fallback parser. The pipeline never breaks because the SLM did. |

---

## 1. MVP definition of done

Running one command:

```bash
python main.py backtest --config configs/default.yaml
```

…downloads (or loads cached) prices for `SPY, TLT, GLD, USO` (+ `^VIX` as a feature), parses the text feed through the SLM layer, fits the factor engine, calibrates five conformal methods, backtests 2020–2025 walk-forward, and writes:

- `reports/backtest_report.md`: summary tables (coverage, width, interval score, VaR tests, latency) and embedded plots
- `reports/figures/*.png`: interval fan charts, rolling coverage, coverage by volatility regime, ACI α-trajectory
- `reports/metrics.json`: machine-readable results (used by tests and the dashboard)

Plus: `pytest` green (including statistical tests of the conformal guarantee), runs offline from a cached/synthetic dataset, and runs without any LLM installed (lexicon fallback). Stretch: a Streamlit dashboard.

### Success criteria (quantitative)

1. On **synthetic exchangeable data**, split conformal coverage across 500 trials lies in `[0.95, 0.95 + 1/(n+1)]` on average (a unit test asserts this).
2. On **real 2020–2025 data**, ACI / rolling conformal achieves empirical coverage within **±1.5 pp of 95%**, and we *report* the naive split conformal shortfall rather than hiding it. **Result: met.** ACI 95.2% (95.3% in 2020), rolling 95.0%; split conformal 93.3% overall and 88.9% in 2020 (numbers as of M5, after the CV-tuned ridge).
3. Normalized/CQR intervals are **narrower on average** than absolute-residual intervals at equal coverage (lower Winkler interval score). **Result: not met on real data.** Winkler (as of M5): Gaussian 2.149, split-absolute 2.162, normalized 2.277, CQR 2.301. The normalized score also under-covered most (91.6%): σ̂ from the log-|residual| model is too noisy to help yet. Revisit with a better scale model in M6. **M6 follow-up:** tried narrower scale models and L1-regularized quantile regression; none beat the absolute residual (best Winkler 2.162 vs CQR-regularized 2.883). Cause: the target is log-vol, whose residuals are already near-homoskedastic. Consequence: online methods now default to the absolute score, which improved ACI (Winkler 2.320 → 2.212, worst quarter 81.0% → 82.5%).
4. Conformal VaR passes the **Kupiec POF test** (p > 0.05) at both 95% and 99%. Measured in M3 on 2020–2026: at 95% the Gaussian, historical-simulation and GARCH baselines all get the breach *rate* right (4.97–5.8%) but all fail the **Christoffersen** clustering test (p ≤ 0.003); at 99% Gaussian (1.92%) and historical simulation (1.69%) fail Kupiec. The bar is therefore 99% coverage and breach clustering, and we report honestly whether ACI reduces clustering.
5. Ablation: the SLM features measurably help in the synthetic world (where the signal is known to exist). On real data we report the result whichever way it goes. **M5 result:** in the synthetic world text helps only with near noise-free headlines (vol RMSE −1.1% at SNR 100 vs −2.2% for the oracle; +0.9% at the default SNR 0.5). On real data, FOMC text hurts with every parser (lexicon +1.2% RMSE; qwen2.5:3b +2.8% after the v2 direction-named schema and training-range clipping, down from +16.7% with v1), so the default config is price-only. See README for the v1→v2 accuracy comparison.
6. Latency: per-day end-to-end scoring (excluding first-time SLM parsing) < 50 ms; SLM parse latency reported per backend (p50/p95). **Measured:** scoring one day p50 0.4 ms; lexicon parse p50 1.9 ms per FOMC statement. SLM (qwen2.5:3b via Ollama, M4 Pro) p50 2.9 s per FOMC statement.

---

## 2. Architecture

```
                 ┌─────────────────────────── data layer ───────────────────────────┐
 yfinance ──▶ PriceStore (parquet cache) ──▶ returns, realized vol, VIX
 FOMC statements / synthetic news ──▶ NewsStore (timestamped, as-of safe)
                 └──────────────────────────────────────────────────────────────────┘
                                   │ text (masked)                  │ prices
                                   ▼                                │
             ┌───────────────────────────────────────┐              │
  C1         │ SLM Semantic Parser                   │              │
             │  backends: ollama | transformers+     │              │
             │  outlines | lexicon (fallback)        │              │
             │  → RiskSignal (pydantic, validated)   │              │
             │  → SQLite cache keyed by content hash │              │
             └───────────────────────────────────────┘              │
                                   │ daily aggregated signals       │
                                   ▼                                ▼
             ┌──────────────────────────────────────────────────────────────┐
  C2         │ Quantitative Factor Engine                                   │
             │  EWMA covariance (λ=0.94) + semantic shock overlay           │
             │  HAR-RV features + signal features → point model μ̂(X), σ̂(X) │
             │  Outputs: σ̂_p (portfolio vol), RV forecast, parametric VaR   │
             └──────────────────────────────────────────────────────────────┘
                                   │ point predictions
                                   ▼
             ┌──────────────────────────────────────────────────────────────┐
  C3         │ Conformal Wrapper (NCM)                                      │
             │  split-abs | split-normalized | CQR | rolling | ACI          │
             │  → C(X_new) = [lo, hi], conformal VaR                        │
             └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
             Backtester (walk-forward) → metrics → report / dashboard
```

---

## 3. Repository layout

```
Aegis/
├── pyproject.toml              # package metadata, ruff + pytest config
├── requirements.txt            # pinned runtime deps
├── requirements-llm.txt        # optional: transformers, outlines, torch
├── README.md                   # quickstart, results table, method notes
├── main.py                     # CLI entry: fetch | parse | backtest | report | dashboard
├── configs/
│   ├── default.yaml            # real data, lexicon parser, all conformal methods
│   ├── synthetic.yaml          # synthetic world, known signal
│   └── ollama.yaml             # real SLM backend
├── src/aegis/
│   ├── __init__.py
│   ├── config.py               # pydantic Settings loaded from YAML
│   ├── data/
│   │   ├── prices.py           # yfinance fetcher + parquet cache + synthetic fallback
│   │   ├── news.py             # NewsItem model, NewsStore, as-of queries
│   │   ├── fomc.py             # real FOMC statement loader (public text)
│   │   └── synthetic.py        # regime-switching market + latent-shock news generator
│   ├── slm_parser.py           # C1: RiskSignal schema, backends, cache, masking
│   ├── features.py             # HAR-RV, EWMA vol, signal aggregation with decay
│   ├── risk_engine.py          # C2: covariance updater, point models, parametric VaR
│   ├── conformal_ncm.py        # C3: all conformal methods (pure numpy, documented math)
│   ├── backtest.py             # walk-forward loop, splits with embargo
│   ├── metrics.py              # coverage, width, Winkler, Kupiec, Christoffersen
│   ├── latency.py              # timing context manager + percentile aggregation
│   └── report.py               # markdown + matplotlib figures
├── dashboard/app.py            # (stretch) Streamlit
├── tests/
│   ├── test_conformal.py       # statistical guarantee tests
│   ├── test_no_lookahead.py    # leakage tests
│   ├── test_slm_parser.py      # schema, fallback, cache determinism
│   ├── test_risk_engine.py
│   └── test_metrics.py         # Kupiec/Christoffersen against known values
├── data/                       # gitignored cache (parquet, sqlite)
└── reports/                    # generated
```

### Dependencies

`requirements.txt` (core, must install cleanly on macOS arm64 / Python 3.12):
`numpy, pandas, scipy, scikit-learn, yfinance, pyarrow, pydantic>=2, pyyaml, matplotlib, tabulate, arch` (GARCH baseline), `lightgbm` (quantile models for CQR), `httpx` (Ollama client), `rich` (console tables), `pytest, hypothesis`.

`requirements-llm.txt` (optional): `transformers, torch, outlines, huggingface_hub`.

Recommended local SLM runtime: **Ollama** (`brew install ollama`), models `qwen2.5:3b-instruct` (fast default), `llama3.1:8b-instruct`, `phi3:mini`. Ollama's `format=<JSON schema>` gives grammar-constrained decoding without the `outlines` dependency stack. It runs on Apple Silicon without CUDA.

---

## 4. Component specifications

### 4.1 Data layer (`data/`)

**Prices (`prices.py`)**
- `fetch_prices(tickers, start="2010-01-01", end=None) -> DataFrame` with adjusted close, cached to `data/prices_{hash}.parquet`. Retries with backoff; on network failure loads cache; if no cache, raises with a pointer to `--synthetic`.
- Derived: daily log returns `r_t`, 5-day forward realized vol `RV_{t→t+5} = sqrt(Σ r²)` per asset and for the portfolio.
- Portfolio: equal-weight by default (configurable weights). Also an inverse-vol variant.

**News (`news.py`, `fomc.py`, `synthetic.py`)**
- `NewsItem(id, published_at: datetime[UTC], source, title, body)`.
- `NewsStore.as_of(t)` returns only items with `published_at < market_close(t)`. This is the *only* accessor the feature builder may use (enforced by test).
- **Real source: FOMC statements** (139 since 2010, public domain, release times parsed from each page, including emergency statements). Downloaded once from federalreserve.gov and cached incrementally to `data/fomc_statements.jsonl`. Offline tests use small hand-written HTML fixtures that mimic both Fed page layouts; no real statement text is bundled.
- **Synthetic source**: a latent macro-shock process `z_t ∈ R^k` (factors: rates, inflation, growth, energy, geopolitics, credit) drives *both* (a) simulated asset returns via factor loadings and a regime-switching volatility, and (b) templated headlines whose wording reflects `z_t` plus noise, at a configurable signal-to-noise ratio. Because we know the truth, we can measure whether the SLM recovers it (correlation of parsed shock vs `z_t`) and whether it improves risk forecasts.
- **Leakage controls**: date/entity masking (`"March 2020"` → `"[DATE]"`, `"COVID-19"` → `"[EVENT]"`) before sending real text to the SLM, configurable on/off so its effect can be measured.

### 4.2 C1 — SLM Semantic Parser (`slm_parser.py`)

**Output schema (pydantic, also exported as JSON Schema for constrained decoding):**

```python
class ShockVector(BaseModel):
    rates: float = Field(0, ge=-1, le=1)
    inflation: float = Field(0, ge=-1, le=1)
    growth: float = Field(0, ge=-1, le=1)
    energy: float = Field(0, ge=-1, le=1)
    geopolitics: float = Field(0, ge=-1, le=1)
    credit: float = Field(0, ge=-1, le=1)

class VolPremiumSignal(BaseModel):
    direction: Literal["up", "down", "neutral"]
    magnitude: float = Field(ge=0, le=1)

class RiskSignal(BaseModel):
    sentiment_score: float = Field(ge=-1, le=1)
    macro_shock_vector: ShockVector
    volatility_premium_signal: VolPremiumSignal
    asset_impacts: dict[Literal["SPY","TLT","GLD","USO"], float]   # each in [-1, 1]
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(max_length=3)   # short quoted spans, aids auditing
```

**Backends** (common `Parser` protocol: `parse(text) -> RiskSignal`, `parse_batch(...)`):
1. `OllamaParser`: HTTP to `localhost:11434/api/chat`, `format=RiskSignal.model_json_schema()`, `temperature=0`, fixed `seed`, few-shot system prompt (3 hand-written examples). Validates; on failure retries once with the validation error appended; then falls back.
2. `TransformersParser`: `outlines` JSON-constrained generation on a HF model (Phi-3-mini / Qwen2.5-1.5B). Optional extra.
3. `LexiconParser` (default, deterministic): Loughran–McDonald-style finance word lists (compact bundled subset) + keyword→factor maps + negation handling. Fast, reproducible, and serves as the **baseline** the SLM must beat.

**Infrastructure:**
- SQLite cache keyed by `sha256(backend, model, prompt_version, masked_text)`. Backtests are re-runnable in seconds, and the cache doubles as an audit log.
- Records per-call latency, tokens, retries, and fallback events into the latency report.
- Daily aggregation (`features.py`): exponentially decayed sum of signals (half-life configurable, default 3 trading days), confidence-weighted, plus `days_since_last_news`.

### 4.3 C2 — Quantitative Factor Engine (`risk_engine.py`)

**Covariance updater**
- EWMA (RiskMetrics): `Σ_t = λ Σ_{t-1} + (1−λ) r_{t-1} r_{t-1}ᵀ`, λ = 0.94.
- **Semantic shock overlay**: `Σ̃_t = D_t Σ_t D_t` with `log σ̃_{i,t} = c_i + β_i log σ_{i,t} + B_i · s_t`, fit per asset by ridge regression of log forward realized vol on `[log σ_EWMA, s_t]` (training split only). β is free because forcing β = 1 generalizes poorly (fitted β ≈ 0.5: EWMA is noisy and vol mean-reverts). The baseline for the ablation is the same overlay fit **without** signals (calibrated EWMA). With oracle shocks in the synthetic world the overlay cuts out-of-sample log-vol RMSE by 1–4% beyond calibrated EWMA; the synthetic world was redesigned in M3 so big news shocks trigger stress regimes that the vol index sees only with a lag (otherwise text had almost no edge to find).
- Portfolio vol `σ̂_p = sqrt(wᵀ Σ̃ w)`; parametric VaR `VaR_α = −(μ̂ + z_α σ̂_p)`.

**Point models μ̂(X)**, predicting the two targets:
- *Target A, log 5-day realized portfolio vol*: features = HAR components (1d, 5d, 22d RV), EWMA σ̂_p, VIX level and change, aggregated SLM signals. Models: `HAR-OLS` (baseline), `Ridge`, `LightGBM` (main). Also LightGBM quantile models at α/2 and 1−α/2 for CQR.
- *Target B, next-day portfolio return*: μ̂ ≈ 0 (honest: daily mean returns are barely predictable), scale σ̂ from the engine. Conformal does the heavy lifting here.

**Baselines for comparison:** Gaussian VaR from EWMA, historical-simulation VaR (250d), GARCH(1,1) via `arch`.

### 4.4 C3 — Conformal Wrapper (`conformal_ncm.py`)

Pure numpy, no ML dependencies, each function with its equation in the docstring and inline comments.

```python
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """q̂ = the ⌈(n+1)(1−α)⌉-th smallest score.
    Equivalent to Quantile(S, ⌈(n+1)(1−α)⌉/n) with 'higher' interpolation.
    Returns +inf when ⌈(n+1)(1−α)⌉ > n (calibration set too small for α)."""
```

| Method | Non-conformity score `S_i` | Interval `C(X_new)` | Why |
|---|---|---|---|
| `SplitAbsolute` (PRD spec) | `|y_i − μ̂(X_i)|` | `μ̂ ± q̂` | The textbook method, kept as reference |
| `SplitNormalized` | `|y_i − μ̂(X_i)| / σ̂(X_i)` | `μ̂ ± q̂·σ̂` | Width adapts to volatility |
| `CQR` (Romano et al. 2019) | `max(q̂_lo(X_i) − y_i, y_i − q̂_hi(X_i))` | `[q̂_lo − q̂, q̂_hi + q̂]` | Asymmetric, heteroskedastic |
| `RollingSplit` | any of the above | recalibrate on last `m` days (default 500) | Tracks regime changes |
| `ACI` (Gibbs & Candès 2021) | any of the above | `α_{t+1} = α_t + γ(α − err_t)` | Long-run coverage → 1−α even under distribution shift |
| `OneSidedVaR` | `−(r_i − μ̂_i)/σ̂_i` | `VaR = −(μ̂ − q̂·σ̂)` | Conformal VaR, lower tail only |

All methods share an interface: `fit(cal_pred, cal_y)`, `predict(pred) -> (lo, hi)`, and `update(y_true)` for online methods. `γ` for ACI defaults to 0.005 with a sensitivity sweep in the report.

### 4.5 Backtester (`backtest.py`)

- **Splits** (default config): train 2010–2017 · calibration 2018–2019 · test 2020-01 → latest. The test window deliberately includes the COVID crash, the 2022 rate shock, and the 2023 bank stress.
- **Embargo** of `h` days (5 for target A) between splits, so overlapping forward-looking targets don't leak.
- **Walk-forward**: at each test day `t`: features from `≤ t` only → point prediction → interval → observe `y` at `t+h` → online update (ACI/rolling) only once `y` is actually known (respects the `h`-day delay).
- Optional periodic model refit (yearly) as a config flag.

### 4.6 Metrics (`metrics.py`)

- Empirical coverage (overall, rolling 63-day, by year, **by volatility tercile**, which tests conditional coverage).
- Average and median interval width; **Winkler/interval score** (rewards narrow *and* correct).
- VaR: breach rate, **Kupiec POF** test (correct breach frequency), **Christoffersen** independence test (breaches not clustered).
- Latency: p50/p95/max per stage (data load, parse, features, predict, conformal), wall-clock per test day.

---

## 5. Report output (example shape)

```
Target A — 5-day realized portfolio volatility, α = 0.05, test 2020-01-02 → 2025-10-31 (n = 1468)

Method              Coverage   Avg width   Winkler   Cov(high-vol tercile)
Gaussian (EWMA)       ...        ...        ...          ...
SplitAbsolute         ...        ...        ...          ...
SplitNormalized       ...        ...        ...          ...
CQR (LightGBM)        ...        ...        ...          ...
ACI + CQR             ...        ...        ...          ...

Target B — 1-day 95% VaR
Method        Breach rate   Kupiec p   Christoffersen p
...

SLM ablation (synthetic world, SNR = 0.5)
Features              RV forecast RMSE   CQR avg width   Shock-recovery corr
prices only           ...
+ lexicon signals     ...
+ SLM (qwen2.5:3b)    ...

Latency (ms)       p50    p95
SLM parse (ollama) ...    ...
Scoring per day    ...    ...
```

No numbers are filled in until the backtest produces them. The report generator writes exactly what was measured.

Figures: (1) fan chart of intervals vs realized vol over the test window with breaches marked; (2) rolling 63-day coverage per method with the 95% line, which shows the naive method dipping in March 2020; (3) ACI's `α_t` trajectory; (4) coverage by vol tercile bar chart; (5) synthetic world: parsed shock vs true latent shock scatter.

---

## 6. Testing strategy

| Test | Asserts |
|---|---|
| `test_conformal_iid_coverage` | 500 trials on iid data: mean coverage ∈ `[1−α, 1−α + 1/(n+1)]` (with Monte Carlo tolerance) for every method |
| `test_quantile_edge_cases` | `n` too small → `inf`; ties; α ∈ {0.01, 0.1}; matches a brute-force sort reference |
| `test_aci_long_run` | Under an abrupt variance shift, ACI's running coverage converges to 1−α; naive split does not |
| `test_no_lookahead` | Perturbing any data after `t` leaves every feature at `t` unchanged (prices and news) |
| `test_embargo` | No calibration target window overlaps the test window |
| `test_slm_schema` | Malformed / out-of-range JSON → validation error → retry → fallback; never raises to caller |
| `test_parser_cache` | Same input → identical output and zero backend calls on second run |
| `test_kupiec_known_values` | Kupiec/Christoffersen statistics match hand-computed textbook examples |
| `test_end_to_end_synthetic` | `main.py backtest --config configs/synthetic.yaml` runs offline in < 60 s and writes `metrics.json` |

Network-dependent tests (yfinance, FOMC download, Ollama) are marked `@pytest.mark.network` / `@pytest.mark.llm` and skipped by default.

---

## 7. Milestones (build order)

Each milestone ends with something runnable and its tests green.

| # | Milestone | Deliverable | Est. |
|---|---|---|---|
| M0 | Scaffold | `pyproject.toml`, requirements, config loader, CLI skeleton, ruff + pytest wired | 0.5 d |
| M1 | **Conformal core first** | `conformal_ncm.py` + `metrics.py` + statistical tests on synthetic iid data. This is the mathematical heart, so it's proven before anything depends on it. | 1 d |
| M2 | Data layer | yfinance fetcher + cache, synthetic market/news generator, NewsStore with as-of guarantees, FOMC loader, no-lookahead tests | 1 d |
| M3 | Factor engine | EWMA covariance, HAR features, point models, baselines (Gaussian, HS, GARCH) | 1 d |
| M4 | Backtester + report v1 | Walk-forward with embargo, all conformal methods on real data with **price-only features**, markdown report + figures | 1 d |
| M5 | SLM parser | Schema, lexicon backend, Ollama backend, cache, masking, signal aggregation, shock overlay; synthetic-world ablation | 1.5 d |
| M6 | Polish | Latency benchmarks, README with results and method notes, `--quick` demo mode, (stretch) Streamlit dashboard | 1 d | **Done:** `demo` (offline, ~3 s), `bench`, Streamlit dashboard, README rewrite, absolute-score default for online methods.

Doing M4 *before* M5 is deliberate. The conformal and risk results stand on their own, so the SLM layer becomes a measured add-on, not a dependency that blocks everything.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| yfinance rate limits / API changes | Parquet cache, retry/backoff, synthetic fallback, pinned version |
| SLM too slow for 15 years of daily text | Real text is sparse (FOMC ≈ 130 docs); synthetic corpus sized to ~1 doc/day; cache makes reruns instant; `qwen2.5:3b` default |
| SLM adds no value on real data | Expected and acceptable. Report it honestly; synthetic world proves the pipeline *can* use signal |
| Coverage looks "too good" | Signals leakage; no-lookahead and embargo tests guard against it |
| Overclaiming guarantees | README "Guarantees & limitations" section stating exactly which coverage statements hold under which assumptions |

---

## 9. What makes this MVP impressive

1. **Honest statistics**: it demonstrates *when* conformal guarantees hold and when they break, then fixes the failure with adaptive methods. That's more convincing than a claimed 95% that silently fails in March 2020.
2. **Real risk-management validation**: Kupiec/Christoffersen tests and comparison against GARCH/historical-simulation baselines are what a risk desk would actually ask for.
3. **Measurable LLM contribution**: the synthetic world with known ground truth turns "the SLM helps" from an assertion into an experiment.
4. **Production hygiene**: constrained decoding, validation and fallback, content-hash caching, latency telemetry, offline reproducibility, a one-command run.

---

## 10. Decisions to confirm before building

1. **SLM runtime**: install Ollama (`brew install ollama && ollama pull qwen2.5:3b`) as the real backend? (Lexicon fallback works either way.)
2. **Portfolio weights**: equal-weight (default) vs inverse-vol?
3. **Dashboard**: include the Streamlit stretch goal in the MVP, or stop at the markdown report?
4. **Environment**: plain `venv` + pip (default) vs installing `uv`?
