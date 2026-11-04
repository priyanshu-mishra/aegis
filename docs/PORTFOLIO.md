# Aegis: write-ups for portfolio, CV and interviews

Four pieces, each ready to copy. Every number comes from the project's saved runs (the
[validation report](MODEL_CARD.md) and `site/public/data`), and the live figures refresh nightly, so
re-check them against the [site](https://priyanshu-mishra.github.io/aegis/) before you paste.

1. [Short summary](#1-short-summary) (the same text is in the README)
2. [Portfolio write-up](#2-portfolio-write-up) (about 900 words)
3. [CV entries in STAR form](#3-cv-entries-in-star-form)
4. [Interview preparation](#4-interview-preparation)
5. [How to improve the project](#5-how-to-improve-the-project)

---

## 1. Short summary

**Thesis.** A risk forecast is only useful if it says how wrong it might be, and the claim has to be
checked rather than assumed. Aegis wraps volatility and Value-at-Risk forecasts in *conformal
prediction* intervals, which promise a stated hit rate without assuming returns follow any
distribution, and audits that promise on 1,712 trading days the model never saw, including the 2020
crash.

**Parts.** (1) A four-ETF portfolio (SPY, TLT, GLD, USO) with prices, the VIX and every FOMC
statement since 2010. (2) A local language model that reads central-bank text. (3) A regression
forecast of volatility and VaR. (4) Conformal intervals, including an adaptive version that corrects
itself after each miss. (5) A walk-forward backtest with no look-ahead. (6) Five classical baselines, regulatory
tests, and position-sizing strategies. (7) A live guided site and a bank-style validation report, both
rebuilt every night.

**Method.** Fit on 2010–2017, calibrate on 2018–2019, then walk forward through 2020–2026 one day at a
time. Adaptive methods see each outcome only after it would really be known (five days later for a
five-day forecast). Every claim is a measured coverage rate or breach rate compared with the
textbook alternative.

**What it demonstrates.** The adaptive intervals held their 95% promise on 95.0% of days (the textbook
method: 93.3%) and 94.5% through 2020 (88.9%). A 99% loss limit was breached on 0.93% of days
against 1.93% for the industry-standard model. Sizing positions on the forecasts halved the worst
fall (−30.2% to −13.6%) and lifted the Sharpe ratio from 0.75 to 0.97. It also shows what failed:
text added nothing, a classical model matched the calibration with less capital, and tail breaches
still cluster.

---

## 2. Portfolio write-up

### Aegis: honest error bars for portfolio risk

*A walk-forward study of conformal prediction for market risk, 2020–2026.
[Live study](https://priyanshu-mishra.github.io/aegis/) · [Code](https://github.com/priyanshu-mishra/aegis)*

**The problem.** Every desk forecasts risk, and those forecasts set position sizes, loss limits and,
at banks, regulatory capital. But a forecast is one number, and the question that decides whether you
can lean on it is *how wrong might it be?* Most models answer that by assuming returns follow a bell
curve, calibrating on calm data, and hoping. In March 2020 the portfolio I studied went from about 7%
volatility to 76% in a few weeks. The most widely used model breached its 99% loss limit on twice as
many days as it promised.

**The idea.** A weather forecaster who says "70% chance of rain" is good if it rains on about 70% of
those days. That property is *calibration*, and it is what a risk model owes its users. Conformal
prediction builds ranges with a stated hit rate from the model's own past errors, with no assumption
about the shape of returns. Its catch is that the guarantee needs the future to resemble the past,
and markets break that. The adaptive version (ACI, Gibbs and Candès 2021) keeps score as it goes:
widen a little after each miss, narrow a little after each hit. Its long-run hit rate is guaranteed
for *any* sequence of outcomes.

**What I built.** A research pipeline and a published study:

- *Data and forecast.* Equal-weight SPY, TLT, GLD and USO, plus the VIX. A regularised regression on
  realised-volatility features forecasts five-day volatility and next-day VaR.
- *Conformal layer.* Split, rolling-window, adaptive (ACI), parameter-free adaptive (DtACI) and
  regime-conditional (Mondrian) conformal intervals, with VaR and Expected Shortfall variants.
- *Five classical baselines.* Gaussian EWMA (RiskMetrics), historical simulation, GARCH, GJR-GARCH and
  filtered historical simulation, so no result stands alone.
- *Language reader.* A 3-billion-parameter open model run locally reads Federal Reserve statements into
  structured risk signals, with a rule-based reader as a transparent baseline.
- *Regulatory tests.* Kupiec and Christoffersen backtests, Acerbi–Székely for Expected Shortfall, the
  Basel traffic light and implied capital.
- *Decisions.* Volatility targeting, loss-limit rules and risk-parity sizing, with costs.
- *Delivery.* An Astro and Svelte site with live in-browser playgrounds, a bank-style validation
  report generated from the data, and GitHub Actions that refetch prices, rerun everything and
  redeploy every weeknight.

**How I kept it honest.** The easiest way to get a good backtest is to let the future leak into the
past, so the design is built around preventing it. Models fit on 2010–2017 only; the interval widths
are set on 2018–2019; nothing from 2020 on informs any parameter. News counts for a day only if it was
published before the 4 p.m. New York close. Adaptive methods learn each five-day outcome five days late.
An end-to-end test changes every price after day *t* and asserts nothing dated on or before *t*
moves. 181 offline tests cover this, the statistics themselves (coverage over 2,000 simulated trials,
the conformal quantile against exact arithmetic) and a check that the browser versions of the
algorithms match Python to the last digit.

**What I found.**

| | Adaptive conformal | Comparison |
|---|---|---|
| 95% interval held, 2020–2026 | **95.0%** of days | textbook split conformal 93.3% |
| Same, in 2020 alone | **94.5%** | textbook 88.9% |
| 99% loss limit breached | **0.93%** of days (target 1%) | RiskMetrics 1.93% |
| Basel green-zone days | **98.8%** | RiskMetrics 56.4% |
| Worst fall, vol-targeted on the conformal bound | **−13.6%** | buy-and-hold −30.2% |
| Sharpe ratio | **0.97** | buy-and-hold 0.75 |

**What did not work, which I report as prominently as what did.**

- Reading the Fed did not help. Adding FOMC text made the volatility forecast 1.2% worse with the
  rule-based reader and 2.8% worse with the language model. About eight statements a year is too sparse a
  signal next to the VIX, which prices the same news within minutes. In a synthetic market where the
  true shocks are known, the pipeline does extract the signal, so the finding is about the data,
  not the method.
- Being calibrated costs capital: 30.6% of portfolio value for the adaptive model against 18.0% for
  RiskMetrics. And filtered historical simulation, a classical model, matched the calibration at
  19.3%. Conformal's real advantage is that it needs no model of the return distribution.
- Tail breaches still cluster, because an adaptive method learns from a miss only after it happens.
- Fancier conformal scores never beat the simplest one.
- I found and fixed four bugs in my own pipeline, each now guarded by a regression test, including
  a metric that silently dropped infinite-VaR days and so flattered the breach rate.

**The language model, precisely.** The text layer uses qwen2.5:3b, run locally, **without
fine-tuning**. Its behaviour is controlled at inference time: output forced to a JSON schema,
fields named for their rising direction, four balanced worked examples, temperature 0 with a fixed seed,
masked dates and entities to block memorised outcomes, validation with one retry, and a rule-based
fallback. Redesigning the schema took sign agreement on synthetic headlines from 72% to 100%.
Fine-tuning was not an option with 139 statements and no labels.

**What this demonstrates.** That I can take a statistical idea from a paper to a validated system:
choose a method for its guarantee rather than its fashion, build the evaluation so it cannot flatter
the method, compare against strong baselines, report negative results, and ship something a
non-specialist can read and a risk reviewer can audit.

*Stack: Python (NumPy, pandas, scikit-learn, LightGBM, arch), Ollama, pydantic, pytest and Hypothesis,
Astro and Svelte, GitHub Actions and Pages.*

---

## 3. CV entries in STAR form

### Full STAR (for a project section or a cover-letter paragraph)

- **Situation.** Standard market-risk models are calibrated on calm data and then meet a crisis. In
  March 2020 the industry-standard model broke its 99% loss limit on twice as many days as it
  promised, which is the failure that raises bank capital charges and triggers model reviews.
- **Task.** Build and independently validate a risk engine whose uncertainty statements hold up on
  data it has never seen, and show whether the promised coverage translates into better decisions.
- **Action.** Implemented conformal prediction (split, rolling, adaptive, DtACI, Mondrian) on top of a
  volatility and VaR forecaster for a four-ETF portfolio. Designed a leakage-proof walk-forward
  protocol (embargoes, delayed feedback, an as-of news clock) and benchmarked five classical models with Kupiec,
  Christoffersen, Acerbi–Székely and Basel tests. Added a locally run language model for Federal
  Reserve text, using constrained decoding and schema redesign. Built volatility-targeting
  strategies, 181 tests, a nightly-refreshed public site, and a bank-style model validation report.
- **Result.** Over 1,712 unseen trading days the adaptive intervals held a 95% target on 95.0% of
  days (textbook method: 93.3%) and 94.5% through the 2020 crash (88.9%); 99% loss-limit breaches
  were 0.93% versus 1.93% for the industry model. Volatility-targeting on the forecasts cut the worst
  drawdown from −30.2% to −13.6% and raised Sharpe from 0.75 to 0.97. I reported the negative
  findings too: text signals added no value, and a classical model matched calibration with less capital.

### Resume bullets (pick two or three)

- Built a conformal-prediction risk engine for a four-ETF portfolio and validated it walk-forward on
  1,712 unseen days: 95.0% interval coverage against a 95% target (textbook method 93.3%), and 94.5%
  through the 2020 crash (88.9%).
- Backtested 99% VaR against five classical baselines with Kupiec, Christoffersen, Acerbi–Székely and the Basel
  traffic light: 0.93% breach rate versus 1.93% for RiskMetrics, in the green zone on 98.8% of days
  versus 56.4%.
- Used the forecasts to size positions: volatility targeting halved the worst drawdown (−30.2% to
  −13.6%) and raised Sharpe from 0.75 to 0.97 after trading costs.
- Engineered for reproducibility: 181 offline tests including look-ahead tests, JS/Python parity checks, an
  SR 11-7-style validation report generated from data, and nightly CI that rebuilds and deploys the
  public study.
- Used a local 3B-parameter LLM with constrained JSON decoding to turn FOMC statements into risk
  signals; fixing the schema raised sign agreement from 72% to 100%. Reported that text did not improve the
  real-data forecast.

### One line (LinkedIn, or a skills-section project title)

*Aegis: conformal risk engine with walk-forward validation, Basel-style backtests and a nightly-updated
public study (Python, LLM, Astro).*

**Don't write:** "fine-tuned an LLM" (it wasn't), "outperformed" without naming the benchmark and
metric, or "predicts the market" (it forecasts risk and its error, not returns).

---

## 4. Interview preparation

| They ask | Short answer |
|---|---|
| *Why conformal instead of GARCH or a bell curve?* | It makes no assumption about the return distribution and gives a coverage guarantee. But the honest result is that filtered historical simulation matched it for less capital: the advantage is generality, not superiority. |
| *Markets aren't exchangeable. Doesn't that break the guarantee?* | Yes for split conformal, and it showed: 88.9% in 2020. ACI drops the assumption: its long-run miss rate is bounded for any sequence, and the floor on its working level keeps it from going infinite. |
| *How do you know there's no look-ahead?* | By construction (embargoes, delayed feedback, 4 p.m. news clock) and by test: perturb every price after day *t* and assert nothing earlier changes. |
| *Did you fine-tune the language model?* | No. Prompting, constrained decoding and schema design. With 139 unlabeled statements, fine-tuning would overfit or memorise outcomes, so I used a synthetic world with known shocks instead. |
| *Why did text not help?* | About eight statements a year against a VIX that reprices the news in minutes. I showed the pipeline does recover the signal when it exists, so the limit is the data. |
| *What's the weakest part?* | One portfolio, one crash, free data, and no confidence intervals on the differences between methods. The tail breaches cluster. Section 5 lists the fixes. |
| *Is it production-ready?* | The validation, monitoring triggers and nightly refresh are. It is not: no intraday data, flat trading costs, one asset class. |

---

## 5. How to improve the project

Ordered by value for effort, each doable in days. Larger research directions are in
[ROADMAP_ADVANCED.md](ROADMAP_ADVANCED.md) and AI-based ones in [AI_EXTENSIONS.md](AI_EXTENSIONS.md).

1. **Put error bars on the comparisons.** Coverage of 95.0% against 93.3% is reported as a point
   estimate. Add block-bootstrap confidence intervals and a Diebold–Mariano-type test for the
   differences. This is the most important statistical gap.
2. **More assets and a second test period.** Add equities in other regions, credit and FX, and a
   pre-2020 test (2008) to see whether the 2020 result generalises.
3. **Realistic costs and capacity.** Replace the flat 5 bp with a spread-and-impact model, and
   report turnover and break-even costs for the sizing rules.
4. **Higher-frequency volatility.** Use intraday realised volatility instead of daily returns, which
   sharpens the point forecast and so narrows every interval.
5. **Fix clustered breaches.** Test a regime-aware widening trigger (Mondrian by VIX level is already
   implemented) and report whether Christoffersen passes at 99%.
6. **Richer text.** The negative result points at *frequency*, not method. Try Fed speeches, minutes
   and headlines, which arrive far more often than statements.
7. **Packaging.** A Docker image, a pinned lockfile and a one-command "reproduce the whole
   study" target make the work easy for a reviewer to run.
8. **Write it up.** A 6–8 page working paper with the results tables would give the project a
   citable form for applications.
