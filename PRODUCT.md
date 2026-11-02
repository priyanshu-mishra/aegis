# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Astro (static) with Svelte islands for interactive pieces and Observable Plot / D3 for charts, deployed to GitHub Pages at `priyanshu-mishra.github.io/aegis`. Data comes from `site/public/data/*.json`, exported by the Python pipeline and refreshed nightly by GitHub Actions (see `site/README.md`). Chosen in `docs/PLAN_v2.md`, approved by the author.

## Users

**Primary:** hiring teams for quant, risk and portfolio roles (portfolio managers, risk managers, quant researchers and the recruiters who screen for them) assessing the author's work. Readers arrive from a CV, LinkedIn or GitHub link and give it a few minutes. Many are not conformal-prediction specialists. They need to understand what was built, why it matters, and whether the rigor is real, quickly.

**Secondary:** technical readers who open the "for practitioners" detail to check methods, math, tests and code.

## Product Purpose

Aegis is a research project and its public write-up. It forecasts portfolio risk (volatility, VaR, Expected Shortfall), wraps the forecasts in conformal prediction intervals, checks whether the stated coverage holds on 2020–2026 market data, applies the calibrated risk to position sizing, and tests whether text signals (FOMC statements, parsed by a lexicon or a local small language model) add anything.

Success for the site: a reader without background can explain what Aegis does and why the 2020 result matters after the plain-language layer, and an expert finds the evaluation rigorous and honest.

## Positioning

The project's distinguishing claim is evaluation honesty, not a single good number. It shows where textbook conformal prediction fails on real markets and how adaptive variants fix it. It backtests like a bank model-risk team would (Kupiec, Christoffersen, Basel traffic light, ES backtest). It reports negative and uncomfortable results as prominently as positive ones: text did not help on real data, filtered historical simulation matches conformal calibration with less capital, and an SLM schema bug was found and fixed.

## Operating Context

- Readers open the site on desktop and mobile, usually once, often between other tasks.
- The site is a companion to the GitHub repository `priyanshu-mishra/aegis`, which holds the code, README, tests and reports.
- Numbers on the site come from the exported JSON, regenerated nightly; the site must show the "data as of" date and whether the latest refresh succeeded (`status.json`).
- Some interactions recompute conformal intervals in the browser from the exported calibration data; they must match Python's reference values (`golden.json`).

## Capabilities and Constraints

- Static hosting only: no server, no database, no user accounts.
- All results are real outputs of the pipeline; the site must not show invented numbers.
- SLM results come from local runs (Ollama is not available in CI); the site uses the committed copies.
- Terminology to keep consistent: coverage, interval, VaR, Expected Shortfall (ES), ACI, DtACI, split conformal, calibration period, test period, breach, Basel traffic light.

## Brand Commitments

- Name: **Aegis**.
- Author byline: **Priyanshu Mishra**, with links to GitHub (`https://github.com/priyanshu-mishra`) and LinkedIn (`https://www.linkedin.com/in/priyanshu-mishra-/`).
- Voice: first-person research note ("I tested…, I found…"), plain language first, honest about what failed.
- License: MIT.

## Evidence on Hand

- All metrics, series and tables: `site/public/data/*.json` (schema in `site/README.md`); human-readable reports in `README.md` and `reports/` after a run.
- Charts and diagrams already produced for the README: `docs/assets/` (light and dark SVG).
- Test suite: 175 offline tests plus network and live-SLM tests; CI on GitHub Actions.
- No testimonials, users, employers, or external endorsements exist; none may be implied.

## Product Principles

1. **Understandable first, rigorous underneath.** Every idea is explained in plain language before the math, and the math is always one click away.
2. **Show the evidence.** Every claim on the page is backed by a chart or number from the exported data.
3. **Failures are findings.** Negative results and fixed bugs are presented as clearly as wins.
4. **Respect a short visit.** The key result is visible within the first screen, and the story can be skimmed by headings and takeaways.

## Accessibility & Inclusion

WCAG 2.1 AA: sufficient contrast, keyboard-operable interactions, text alternatives (summary or table) for every chart, and support for reduced motion.
