# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Aegis is a conformal risk engine for a four-ETF portfolio (SPY, TLT, GLD, USO). It forecasts volatility and VaR/ES, wraps the forecasts in conformal intervals, and backtests them walk-forward on 2020–2026. It has two halves: a Python research pipeline (`src/aegis`) and a static Astro site (`site/`), published at https://priyanshu-mishra.github.io/aegis/. The site is built from the pipeline's exported JSON.

## Commands

Python, from the repo root. Use the project venv: `.venv/bin/python` (the system `python3` does not have `aegis` installed).

```bash
pip install -e ".[dev]"                                # extra: [dashboard]
ruff check . && ruff format --check .                  # CI runs both; line length 100
pytest -q                                              # offline suite (network/llm markers deselected by default)
pytest tests/test_conformal.py::test_name -q           # single test
pytest -m network                                      # yfinance / federalreserve.gov tests
pytest -m llm                                          # needs Ollama with qwen2.5:3b
```

Pipeline (each step reads the previous step's output under `reports/<config name>/`):

```bash
python main.py demo                                    # offline end-to-end on a synthetic world
python main.py fetch -c configs/default.yaml           # cache prices + FOMC statements into data/
python main.py backtest -c configs/default.yaml
python main.py ablation -c configs/default.yaml        # with/without text (also run on configs/synthetic.yaml)
python main.py sensitivity -c configs/default.yaml     # α, γ, window, λ grid (--quick skips λ re-runs)
python main.py export-site -c configs/default.yaml     # writes site/public/data/*.json
python main.py model-card -c configs/default.yaml      # writes docs/MODEL_CARD.md from site/public/data
python scripts/make_readme_assets.py                   # regenerates README SVGs from saved runs
```

Config values can be overridden with `--set conformal.alpha=0.1` (repeatable). Configs are pydantic-validated in `src/aegis/config.py`, so a misspelled key fails.

Site, from `site/`:

```bash
npm ci
npm run dev        # http://localhost:4321/aegis/ (base path is /aegis)
npm test           # JS ports vs Python golden files (node --test, no framework)
npm run build
```

## Architecture

**Pipeline flow** (`backtest.py` orchestrates):
- `data/`: prices, FOMC news, and a synthetic world with known shocks.
- `features.py`: HAR, EWMA and VIX features, plus as-of-safe text signals.
- `risk_engine.py`: EWMA covariance and the point model (ridge by default).
- `conformal_ncm.py`: non-conformity scores and the split, rolling, ACI, DtACI and Mondrian methods, run online by `run_online` with a feedback lag.
- Baselines and tail measures: `baselines.py` (Gaussian, historical simulation, GARCH, GJR, FHS) and `risk_measures.py` (ES, Acerbi–Székely Z2, Basel traffic light, capital, Euler contributions).
- `portfolio.py`: sizing strategies.
- `metrics.py`: coverage, Winkler, Kupiec and Christoffersen.

**Text layer.** `slm_parser.py` turns text into a `RiskSignal` using either the lexicon or Ollama. Ollama output is schema-constrained, retried, and falls back to the lexicon. Text is **off** in `configs/default.yaml` (`parser.use_signals: false`) because it did not help on real data.

**Leakage rules are the core invariant.** Splits are train ≤2017 / calibration 2018–2019 / test 2020+, with a 5-day embargo. Online methods only see an outcome after `feedback_lag` (= horizon) days. News counts for a day only if published before the 4 p.m. New York close. Many tests in `tests/test_no_lookahead.py` and elsewhere perturb future data and assert nothing earlier changes. Any new feature or method must preserve this and get such a test.

**Site data contract.** `export.py` writes `site/public/data/*.json` (schema in `site/README.md`): columnar series of the form `{dates, series}`, plus `metrics.json`, `meta.json` and `status.json`. VaR breach flags (`<method>_breach`) are exported at full precision because the exported values are rounded. The site reads these files at build time through `site/src/lib/data.ts`. **Never hard-code a number in the site, README figures or the model card**; every figure must come from this data. `BreachStrip` throws at build time if its counts disagree with `metrics.json`.

**Python ↔ browser parity.**
- `site/src/lib/conformal.ts` re-implements split conformal and ACI for the interactive playgrounds. It must match `golden.json` (exported by `export.py`) to 1e-9.
- `site/src/lib/lexicon.ts` ports the lexicon parser and must match `site/tests/lexicon_golden.json`.
- When you change conformal or lexicon logic in Python, update the TS port and regenerate the golden files (`export-site`; `python scripts/make_lexicon_golden.py`). `tests/test_lexicon_golden.py` fails if the lexicon file is stale.

**Generated artifacts with staleness tests.**
- `docs/MODEL_CARD.md` is rendered by `src/aegis/model_card.py`. `tests/test_model_card.py` fails if it differs from a fresh render of the committed data, so rerun `model-card` after committing new exported data.
- `site/src/lib/repo.ts` counts test functions in `tests/` at build time for the site's copy.

**Site structure.** Astro 5 + Svelte 5 islands (`site/src/islands`, `client:visible`). Charts are build-time SVG drawn with d3-scale and d3-shape (`site/src/lib/chart.ts`), each rendered twice (wide and narrow, toggled at 600px). One page, `pages/index.astro`, composes the chapters in `site/src/chapters/`; the chapter list lives in `lib/chapters.ts`. Colours are all tokens in `styles/tokens.css`; dark mode is a second token set under `prefers-color-scheme` and `[data-theme]`, set before paint by an inline script in `Base.astro` and flipped by the toggle in `RunningHead.astro`; `ProgressBar.astro` draws the reading-progress line. Visual rules ("The Prospectus": paper and ink, a single red for misses and breaches, lettered exhibits each with a table view) are documented in `DESIGN.md`; product and audience context is in `PRODUCT.md`.

## CI and deployment

- `.github/workflows/ci.yml` runs ruff and the offline pytest suite on Python 3.11 and 3.12.
- `.github/workflows/site.yml` runs on weeknights (and on pushes to site, src, tests or config). It refetches data, reruns backtest, ablation, sensitivity and export, runs `npm test`, then builds and deploys to GitHub Pages. A final `lighthouse` job audits the live page on mobile (median of 3 runs; `scripts/lighthouse_check.mjs`: performance ≥ 90, other categories 100) without blocking the deploy.
- If the refresh fails, the workflow deploys the committed `site/public/data` snapshot and `status.json` records `fresh: false`. Nightly data is not committed back.

## Conventions

- Commit with `git -c user.name="Priyanshu Mishra" -c user.email="priyanshum.info@gmail.com"`. Do not change the global git config.
- `git push` can hang on the macOS keychain helper; use `GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push`.
- Site and README copy is a first-person research note in plain language. Negative results and fixed bugs are reported as prominently as wins.
