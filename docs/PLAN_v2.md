# Aegis v2: plan

Three goals:

1. **Make the project speak to quant and portfolio roles.** Turn "calibrated intervals" into the language a risk desk or portfolio manager uses: Expected Shortfall, Basel backtesting, risk decomposition, and sizing positions with calibrated risk.
2. **Publish a live, guided site.** A public page that takes a reader with no background from "why does this matter?" to the full results, and updates itself every night. No localhost.
3. **Replace the generic UI.** A distinctive editorial design with a clear narrative flow. Simple on the surface, full detail one click deeper.

Status: **V1, V2 and V3 done.** V3: FHS and GJR-GARCH baselines, DtACI (with issued-level scoring for delayed feedback), Mondrian conformal by volatility regime, and a sensitivity grid over α, γ, window and λ. V1: ES 97.5% with the Acerbi–Székely backtest, 10-day 99% VaR, Basel traffic light and capital, Euler risk decomposition. V2: vol targeting (EWMA / point / conformal upper bound, scales fitted on the calibration period), VaR-limit sizing, equal-weight / inverse-vol / risk-parity allocations, costs and performance metrics. V4: `export-site` (columnar JSON, golden values for the JS port, lexicon rules), data contract in `site/README.md`, and a nightly `site-data` workflow with fallback to the committed snapshot. Next: V5 design system. Estimates assume focused working days.

---

## Part A · Quant and portfolio extensions

Today Aegis answers *"is the risk forecast calibrated?"*. The extensions answer the questions a hiring manager asks next: *"so what does it change for a portfolio, and would it pass model validation?"*

### A1. Risk measures a desk actually reports (high priority)

| Addition | Why it matters for the role | Implementation |
|---|---|---|
| **Expected Shortfall (ES 97.5%)** | Basel III/FRTB replaced VaR with ES for market-risk capital. Interviewers expect it. | Conformal ES: the average of calibration scores beyond the VaR quantile. Baselines: Gaussian, historical simulation, GARCH-t. Backtest with the Acerbi–Székely Z₂ test. |
| **10-day horizon** | The regulatory horizon for VaR and ES. | Add `horizon_days=10` to target B, with the same walk-forward machinery and feedback delay. |
| **Basel traffic-light test** | How regulators actually grade VaR models: green/yellow/red zones by exceptions in 250 days, which sets the capital multiplier (3.0 → 4.0). | Rolling 250-day exception counts → zone timeline and the implied capital multiplier per method. A very legible chart for non-experts. |
| **Risk decomposition** | Portfolio managers ask "which position is driving risk?" | Component VaR and marginal contribution to risk (Euler allocation) from the engine's covariance, shown per asset over time. |
| **Stress scenarios** | Standard in risk reports. | Replay COVID (Feb–Apr 2020), the 2022 rate shock and the 2023 bank stress; compare predicted bands against realized losses. |

### A2. Use the calibrated risk to make decisions (high priority)

This is the bridge from risk modelling to portfolio management, and the part most likely to impress.

- **Volatility-targeted portfolio.** Scale exposure to hit 10% annualized volatility using (a) EWMA vol, (b) the point forecast, and (c) the **conformal upper bound** (a "risk-aware" sizing rule). Report Sharpe, Sortino, max drawdown, how often realized vol overshoots the target, turnover and transaction-cost-adjusted returns.
- **VaR-limit rule.** Cut exposure whenever the 99% conformal VaR exceeds a limit; count limit breaches and the return given up.
- **Allocation comparison.** Equal-weight vs inverse-vol vs risk parity (equal risk contribution), each risk-managed by the engine.
- The expected story: *calibrated uncertainty gives fewer target overshoots and shallower drawdowns at similar Sharpe*. If it doesn't, we report that, as before.

### A3. Model validation framing (medium priority)

- A **model validation report** (`docs/MODEL_CARD.md`) structured like an SR 11-7 / bank model-risk review: purpose, data, assumptions, limitations, performance, ongoing monitoring, and what would trigger a recalibration.
- **Champion vs challenger:** ACI as champion and the Gaussian EWMA as incumbent, with a formal comparison table.
- **Sensitivity analysis:** coverage and width as functions of α, γ, calibration-window length and EWMA λ, shown as small multiples.

### A4. Stronger methods and baselines (medium priority)

- **Baselines:** filtered historical simulation (FHS: GARCH-filtered residuals with bootstrapping), which is the strongest classical VaR baseline, plus GJR-GARCH for leverage effects.
- **Conformal:**
  - DtACI, the parameter-free version of ACI (Gibbs & Candès 2022), so γ needn't be tuned by hand.
  - Conformal PID control (Angelopoulos et al. 2023).
  - **Mondrian (regime-conditional) conformal**, calibrating separately by VIX tercile. This targets the conditional-coverage gap the current report already shows.
- **A larger universe:** 11 sector SPDR ETFs, adding a cross-sectional result (do the guarantees hold per asset?) with joint coverage via a max-score.

### A5. Text layer, honestly scoped (low priority)

- Denser free sources: Fed speeches, FOMC minutes (released three weeks after each meeting, so their timing must be modelled), and press-conference transcripts. Roughly 10× more documents than statements alone.
- An **event study**: realized vol and interval width around FOMC days. This is where text should matter most.
- Keep the honest negative result as a feature: the synthetic SNR sweep stays the centerpiece of this section.

### A6. Engineering credibility (runs alongside)

- A nightly data refresh and backtest in GitHub Actions (see Part B). The site shows "last updated" and live running coverage.
- Versioned run outputs (`runs/<date>/`), so the site can show how a live forecast compared with what happened.
- Make `pip install` reproducible: a lock file (`uv`) and an optional Dockerfile.

---

## Part B · Live site and deployment

### Recommended architecture: a static site with a nightly data pipeline

```
GitHub Actions (cron, nightly)                          GitHub Pages
┌─────────────────────────────────────────────┐        ┌──────────────────────────┐
│ 1. fetch prices (yfinance) + FOMC (cached)   │        │ Astro static site        │
│ 2. run backtest + extensions (Python)        │  ───▶  │ + interactive islands    │
│ 3. `aegis export-site` → compact JSON        │ deploy │ + JSON data (~300 KB gz) │
│ 4. build site, deploy to Pages               │        │ priyanshu-mishra.github  │
│    (on failure: keep last good data)         │        │   .io/aegis              │
└─────────────────────────────────────────────┘        └──────────────────────────┘
```

**Why static rather than hosting the Streamlit app:**
- **Reliability:** free Streamlit hosting sleeps when idle and takes a while to wake; a static site is instant and never goes down.
- **Design:** Streamlit's look is the main source of the "drab" feel and can't be fully redesigned.
- **Interactivity still works:** the interactive parts can run in the browser. The site ships calibration scores and test forecasts (about 1,700 rows), so the browser can recompute split conformal and ACI live as the reader drags α or γ. Real data, no server.
- **Cost:** free, nothing to maintain, and a custom domain is optional.

**The pipeline:**
- `aegis export-site` (new CLI command) writes `site/public/data/*.json`: per-day series, metrics, ablations, ACI paths, calibration scores, the portfolio backtests from Part A, and run metadata.
- `.github/workflows/site.yml`:
  - **Triggers:** pushes to `main`, plus a daily cron at 22:30 UTC (after the US close).
  - **Steps:** Python build, then Node build, then `actions/deploy-pages`.
  - **Failure handling:** if the data download fails, it deploys the last committed snapshot and shows a "data as of" date.
  - **SLM:** Ollama is not run in CI. SLM results come from the committed cache, or the page says they're stale.
- Lighthouse and link checks run in CI. The build fails below a performance or accessibility score of 95.

**Alternatives considered:**
- Streamlit Community Cloud: too slow to wake and too limited in design.
- A Vercel/Next.js server: unnecessary for this data.
- Hugging Face Spaces: same problems as Streamlit.

The Streamlit dashboard stays in the repo as a local "analyst console"; the site replaces it publicly.

### Site tech stack

| Layer | Choice | Reason |
|---|---|---|
| Framework | **Astro** + MDX | Content-first, ships zero JavaScript by default, and adds interactive "islands" only where needed; good for a long guided page |
| Islands | **Svelte** | Small, fast components for the sliders and playgrounds |
| Charts | **Observable Plot** (+ D3 for custom pieces) | Expressive, small, and themeable with our color tokens; exportable as SVG |
| Math | KaTeX (rendered at build time) | Fast and no layout shift |
| Scroll steps | IntersectionObserver (no heavy library) | Keeps the bundle small |
| Fonts | self-hosted subsets | No layout shift; works offline |

---

## Part C · Design revamp: from dashboard to guided story

### C1. Design direction: an editorial research note, not a SaaS dashboard

What "AI slop" looks like, and what we'll do instead:

| Avoid | Do instead |
|---|---|
| Purple/indigo gradients, glassmorphism, glowing blobs | A warm paper background (`#f7f5f0`) and near-black ink, with a deep slate for dark mode |
| Grids of identical rounded cards with emoji icons | A single column of reading text with **margin notes** (Tufte-style sidenotes) and full-bleed charts |
| Generic sans-serif everywhere | **A serif display face** (e.g. Newsreader or Source Serif 4) for headlines, a humanist sans (IBM Plex Sans) for body text, and **IBM Plex Mono with tabular figures** for every number |
| Many accent colors | **One accent color** (signal blue) for "the model" and **one alarm color** (red) only for misses and breaches. Everything else in grays. |
| Decorative animation | Motion only where it explains something: a band widening after misses, a quantile line sliding as α changes |
| Number-heavy hero sections | A hero that asks a question, with a single live chart |

The aim is a page that reads like a well-designed research note or a Financial Times / Bloomberg Graphics explainer. During the build, the installed `impeccable` and `design-taste-frontend` skills can supply design direction and critique passes, and the `dataviz` skill keeps the charts consistent.

### C2. Narrative structure (the "flow")

Each chapter has **one idea, one main visual and one takeaway**. Every chapter has two levels: a plain-language layer always visible, and an expandable **"For practitioners"** drawer with the math, code links and full tables. Technical terms carry hover definitions that link to a glossary.

| # | Chapter | Plain-language question | Main visual / interaction |
|---|---|---|---|
| 0 | **Hero** | "How sure is your risk model?" | A live band drawing itself over 2020; one sentence; scroll cue |
| 1 | **Why this matters** | Forecasts without honest error bars cause real losses | The weather-forecast analogy ("does 70% rain happen 70% of the time?") leading into March 2020, when standard VaR models broke |
| 2 | **Meet the portfolio** | What are we forecasting? | Four assets, explained in one line each; an interactive price and volatility chart with hover |
| 3 | **Forecasting risk** | How do you predict next week's volatility? | Step-through: yesterday's volatility, then a smoothed average (EWMA), then a learned model; error shrinks at each step |
| 4 | **Adding honest error bars** | How do you turn a guess into a guarantee? | **Interactive conformal playground**: a histogram of past errors, an α slider moves the cut-off, and the band updates on the real test set with coverage shown live |
| 5 | **When the world changes** | Why do guarantees break, and how do we fix them? | Scroll-driven 2020 replay: split conformal misses, ACI adapts; a γ slider re-runs ACI in the browser |
| 6 | **Value-at-Risk, the bank's view** | What's the worst day we should expect? | A breach timeline, a Basel traffic-light timeline (A1), and the capital multiplier per method |
| 7 | **Putting it to work** | Does it improve a real portfolio? | The vol-targeting backtest (A2): equity curves, drawdowns and target overshoots, with a toggle between sizing rules |
| 8 | **Can words help?** | Can the model read the news? | A parser demo: pick a headline, see the parsed shocks (the lexicon is ported to the browser for live input; SLM output is precomputed); the SNR curve; the schema-bug story |
| 9 | **How we know it's honest** | Could it be cheating? | An animated walk-forward timeline, the leakage-test explainer, and the test suite |
| 10 | **Scorecard** | So what's the verdict? | One summary table (verdict per question), "what didn't work", and limitations |
| 11 | **Appendix** | Everything else | Math, methods, glossary, a reproduce-it-yourself section, and repo links |

Navigation: a slim chapter progress rail (sticky on desktop, a top bar on mobile); every chapter has an anchor link to share; reading time about 12 minutes on the plain-language layer.

### C3. Component inventory

- `Chapter` (number, title, question, takeaway), `Sidenote`, `Term` (hover definition), `PractitionerDrawer`, `Callout` (with only two styles: insight and caveat), `StatInline` (monospace number with a delta).
- **Charts:**
  - line or band with a hover crosshair
  - histogram with a draggable quantile
  - small multiples
  - traffic-light timeline
  - equity curve with drawdown underlay
  - heatmap
  - dot plot
- **Playgrounds:**
  - `ConformalPlayground` (α)
  - `ACIPlayground` (γ, score type)
  - `ParserDemo` (headline input)
  - `SizingToggle` (vol-target rule)
- **Shared design tokens** (CSS custom properties) for color, type scale and spacing, with light and dark themes. Charts read the same tokens, so the site and the README stay consistent.

### C4. Quality bars

- **Performance:** Lighthouse 95 or better in every category; under 200 KB of JavaScript on first load; data loaded lazily per chapter.
- **Accessibility:** WCAG AA contrast; every chart has a text summary and a data-table fallback; keyboard-operable sliders; motion disabled for readers who prefer reduced motion.
- **Mobile:** single column at 360 px, charts re-flow, and playgrounds stay usable by touch.
- **Comprehension test:** a reader with no finance background can answer "what does the project do, and why does the 2020 chart matter?" after the plain-language layer. Check this with 2–3 people before launch.

---

## Part D · Milestones

| # | Milestone | Deliverable | Est. |
|---|---|---|---|
| V1 | **Quant core** ✅ | ES 97.5% plus its backtest, 10-day horizon, Basel traffic light, component VaR. Also found and fixed unbounded ACI VaR at extreme quantiles (α floor, γ/h for overlapping targets) and a metrics bias that dropped infinite VaR days. | 2 d |
| V2 | **Portfolio application** ✅ | Vol-targeting and VaR-limit backtests, risk parity, transaction costs, metrics. Result: vol targeting halves drawdown; conformal sizing adds a small edge; conformal VaR limits are breached 0.82% of days vs 1.81% with Gaussian VaR; risk parity did not help. | 2 d |
| V3 | **Methods** ✅ | FHS and GJR-GARCH baselines, DtACI, Mondrian conformal, sensitivity grid. FHS matches conformal calibration with less capital; DtACI gives the most even regime coverage; Mondrian fixes high-vol coverage but its cut-offs drift; ACI hits target at every α and across a 50× range of γ. | 1.5 d |
| V4 | **Data export and pipeline** ✅ | `aegis export-site`, JSON schema, nightly Actions workflow, fallback to last good data. 22 files, 1.9 MB raw / 0.43 MB gzipped, loaded per chapter. | 1 d |
| V5 | **Design system** ✅ | "Prospectus" direction: white paper, black ink, one red for misses and breaches; Source Serif 4; lettered exhibits with table views; DESIGN.md. Charts are build-time SVG (d3-scale/shape) drawn wide and narrow. | 1.5 d |
| V6 | **Story chapters 0–5** ✅ | Cover to ACI, with split-conformal and ACI playgrounds that share one α and match golden.json exactly in the browser. | 2.5 d |
| V7 | **Story chapters 6–11** ✅ | Tail risk, Basel view, sizing, a live lexicon parser (matches Python), risk factors, and how it was checked. Breach flags now exported at full precision. | 2.5 d |
| V8 | **Publish** ✅ | Live at https://priyanshu-mishra.github.io/aegis/. The `Site` workflow refreshes data each weeknight, runs the JS parity tests, builds and deploys; it falls back to the committed snapshot if the download fails. Lighthouse (mobile): 100 in all four categories on the live site (median of 3 runs); a post-deploy `lighthouse` job now checks it nightly. | 0.5 d |
| V9 | **Polish and review** ✅ | Model validation report (A3) at `docs/MODEL_CARD.md`, generated from the exported data by `python main.py model-card` with a staleness test; champion vs incumbent, sensitivity, limitations and monitoring triggers. Mobile pass (no sideways scroll at 375 px, tap targets of 24 px or more), reduced motion already honoured, copy edits for three misleading sentences. | 1 d |

Total ≈ 14.5 days. **Suggested order:** V1 → V2 first (the substance), then V4 → V5 → V6–V8 (the site), then V3 and V9. The site can launch with chapters 0–5 and 9–11 while V2's chapter is still being finished.

## Part E · Risks

| Risk | Mitigation |
|---|---|
| yfinance blocked or rate-limited in CI | Deploy the last committed data snapshot; show a "data as of" date; optionally add a Stooq fallback |
| Live results drift from the README numbers | README charts are regenerated from the same exported JSON, nightly or weekly |
| In-browser recomputation differs from Python | Golden-file test: the JS split conformal and ACI must match Python output to 1e-9 on the exported data |
| The vol-targeting result is unimpressive | Report it honestly; the strength of the project is the evaluation discipline, not one good number |
| Scope creep in the design | Each chapter is shippable on its own; launch with fewer chapters rather than an unfinished whole |
| Fonts and design are subjective | One design-direction checkpoint (a static mock of chapters 0–1) before building everything else |

## Part F · Decisions needed before starting

1. **Domain:** `priyanshu-mishra.github.io/aegis` (free, default) or a custom domain?
2. **Scope of the first release:** everything above (about 15 days), or launch the site on the existing results first (V4–V8, about 8 days) and add the quant extensions afterwards?
3. **Universe:** keep the four-asset portfolio for clarity, or move to 11 sector ETFs for a more realistic portfolio case?
4. **Design direction:** approve the editorial research-note look above, or review two alternative static mocks first?
