# Taking Aegis to master's-thesis scope

Aegis is a strong applied project: one portfolio, one crash, a careful pipeline. A master's thesis
needs one more thing, a **research question whose answer is not already known and whose evidence would
convince a sceptical reader**. This document turns the project's loose ends into thesis-sized
questions. Three are worked through in detail, the rest are sketched.

Citations are from memory of the literature. Check every one before using it in a proposal.

## What the current project gives you, and what it lacks

| Already in place | Missing for a thesis |
|---|---|
| A leakage-proof walk-forward harness, with tests | Statistical inference on the comparisons (confidence intervals, multiple-testing control) |
| Online conformal methods (split, rolling, ACI, DtACI, Mondrian) | Theory beyond reproducing published guarantees |
| Five classical baselines and the regulatory tests (Kupiec, Christoffersen, Acerbi–Székely, Basel) | More than one asset set, one test period, one crisis |
| A synthetic world with known ground truth | A claim that is new, not a replication |

The single most valuable upgrade, before choosing a topic, is **inference**: block-bootstrap confidence
intervals for coverage and breach rates, and a model-confidence-set or superior-predictive-ability test
over the methods. Every direction below needs it.

---

## Direction 1: a large-scale empirical study of online conformal risk

**Question.** Across many assets and many crises, when do adaptive conformal methods deliver calibrated
VaR, how does that depend on the method's tuning, and does any of it beat classical models once you
account for statistical noise?

**Why it is thesis-sized.** Published conformal-for-time-series work mostly uses a few datasets and
reports point estimates. A careful, large, well-controlled empirical comparison is a legitimate and
useful contribution, and the Aegis harness is most of the infrastructure.

**Method.**
- Universe: several hundred liquid assets (equity indices, single stocks, bonds, FX, commodities, crypto
  as a stress case), daily, 2000 to the present, with a pre-registered list of crisis windows (2008,
  2011, 2015, 2020, 2022).
- Methods: ACI, DtACI, agnostic ACI (Zaffran et al., 2022), EnbPI (Xu and Xie, 2021), weighted
  conformal under non-exchangeability (Barber et al., 2023), against FHS, GARCH-family and
  historical simulation.
- Inference: block bootstrap, the Model Confidence Set (Hansen, Lunde and Nason, 2011) or SPA test, and
  a multiple-testing correction across assets.

**Evaluation.** Coverage and breach rates with intervals; clustering via Christoffersen and a
duration-based test; efficiency (width or capital) at equal coverage; behaviour in the first days of a
crisis; sensitivity to step size and feedback delay.

**Risks.** Compute and data cleaning dominate; results may be "no method dominates", which is a valid
finding if the inference is sound.

**Deliverable.** A reproducible benchmark and a paper-shaped result. About 4 to 6 months.

---

## Direction 2: conformal risk control for position sizing

**Question.** Aegis sizes positions on a calibrated risk forecast and reports better drawdowns, but with
no guarantee on the *decision*. Can the sizing rule itself carry a distribution-free guarantee, such as
a bound on expected shortfall relative to a target, or on the chance of exceeding a drawdown limit?

**Background.** *Conformal risk control* (Angelopoulos, Bates, Fisch, Lei and Schuster, 2022) calibrates a
parameter so that the expected value of a monotone loss is controlled. Volatility targeting has exactly
such a parameter: the leverage scale. Today it is fitted on the calibration period.

**Method.**
- Define the loss (for example, the realised excess of next-period loss over a limit) and show it is
  monotone in the leverage parameter.
- Calibrate the leverage online with delayed feedback, adapting the ACI-style update to risk control.
- Compare against the current vol-targeting rules and against robust or CVaR-constrained
  optimisation.

**Evaluation.** Realised violation rates against the target with confidence intervals; return given up
for the guarantee (the efficiency frontier); behaviour through 2020; transaction costs.

**Risks.** Guarantees for non-monotone losses and under serial dependence are subtle, so some theory
will need care or honest relaxation. The empirical story is cleaner than the theory.

**Deliverable.** A new decision rule with a stated guarantee, tested on real data. About 5 to 7 months,
and the one with the clearest "new" claim.

---

## Direction 3: look-ahead-safe language-model signals, with sim-to-real fine-tuning

**Question.** Can a language model's reading of central-bank and news text improve a volatility
forecast *after* controlling for the fact that the model may already know the future, and does
fine-tuning on a synthetic, labelled world transfer to real text?

**Background.** Aegis found text added nothing on eight statements a year. Two things change: more text
(Fed speeches, minutes, headlines), and a rigorous treatment of look-ahead bias, a live research area
(for example Glasserman and Lin, 2023; Lopez-Lira and Tang, 2023 on LLM return prediction).

**Method.**
- Build a look-ahead audit: tests that probe whether the model recalls post-date outcomes, and a
  restriction to text published after the model's training cutoff.
- Compare rule-based, prompted, embedding-based and fine-tuned readers (LoRA on a 0.5B to 1.5B model,
  trained on the synthetic world's unlimited labelled headlines).
- Keep the synthetic positive control and the unchanged price-only baseline.

**Evaluation.** Forecast-error change versus price-only with confidence intervals; sim-to-real gap
(synthetic held-out versus real); latency; the audit's pass rate.

**Risks.** Data access (licensed headlines), sim-to-real failure, and the real possibility that the honest
answer remains "no improvement". The audit methodology is the contribution either way.

**Deliverable.** A methodology for testing LLM signals without look-ahead, plus a fine-tuned model.
About 5 to 8 months.

---

## Other directions, in brief

| Idea | Question | Hook | Difficulty |
|---|---|---|---|
| **Conformal Expected Shortfall** | Can ES be calibrated directly, using the joint elicitability of (VaR, ES) (Fissler and Ziegel, 2016)? | Acerbi–Székely Z2 is already implemented; the current ES method is simple | Medium to high; theory-heavy |
| **Portfolio-level guarantees for all weights** | A VaR band valid for every portfolio weight at once, not one fixed portfolio | Multivariate conformal for covariance or returns | High |
| **Conditional coverage** | Marginal coverage can hide bad regimes; what is achievable? Distribution-free conditional coverage is impossible in general (Foygel Barber et al., 2021), so which relaxations matter in finance? | Mondrian conformal by VIX regime is built | Medium |
| **Learned regime signals** | Do early-warning models reduce first-day breaches (see [AI_EXTENSIONS.md](AI_EXTENSIONS.md), B)? | Needs the inference upgrade first | Medium |
| **Pretrained time-series models** | Does a foundation model give narrower intervals at equal coverage (A in the AI document)? | Check training-data overlap | Medium |
| **Streaming and intraday** | A real-time engine with latency guarantees on intraday realised volatility | Aegis scores one day in 0.39 ms already | Medium; engineering-heavy |
| **Deep hedging with a risk constraint** | Train a hedging policy under a conformal loss constraint (Buehler et al., 2019, for the base method) | Pairs with Direction 2 | High |

## How I would choose

- **Optimise for a defensible result, not a favourable one.** Direction 1 cannot fail (a null result is a result);
  Direction 2 has the clearest novelty; Direction 3 is the most topical and the most fragile.
- **Fit to the programme.** A statistics or maths master's suits 2 (theory) or 1 (inference). A computer
  science or ML master's suits 3. A quantitative finance master's suits 1 or 2.
- **Budget the inference upgrade first.** Two to three weeks on bootstrap intervals and a confidence set
  pays off in every direction.

## A realistic plan (about nine months)

| Months | Work |
|---|---|
| 1 | Inference upgrade on the existing study. Literature review. Pick the direction. |
| 2 to 3 | Reproduce the key published results inside the harness. Extend the data. |
| 3 to 6 | Core method and experiments. Write as you go: the methods section first. |
| 6 to 8 | Robustness, ablations, negative results, the second market and crisis. |
| 8 to 9 | Draft, advisor feedback, a public reproducible release. |

## Reading list to start

- Vovk, Gammerman and Shafer, *Algorithmic Learning in a Random World* (2005): the foundations.
- Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification* (2021): the accessible overview.
- Gibbs and Candès, *Adaptive Conformal Inference Under Distribution Shift* (2021), and *Conformal
  Inference for Online Prediction with Arbitrary Distribution Shifts* (2022/2024): what Aegis implements.
- Zaffran, Féron, Goude, Josse and Dieuleveut, *Adaptive Conformal Predictions for Time Series* (2022).
- Barber, Candès, Ramdas and Tibshirani, *Conformal Prediction Beyond Exchangeability* (2023).
- Angelopoulos, Bates, Fisch, Lei and Schuster, *Conformal Risk Control* (2022).
- Christoffersen (1998) and Kupiec (1995) on VaR backtesting; Acerbi and Székely (2014) on ES backtesting;
  Fissler and Ziegel (2016) on elicitability of (VaR, ES).
- Gu, Kelly and Xiu, *Empirical Asset Pricing via Machine Learning* (2020): the ML-in-finance benchmark.
- Venues to aim a paper at: the Symposium on Conformal and Probabilistic Prediction (COPA), workshops
  at NeurIPS and ICML on conformal and time-series methods, and the *Journal of Risk* or
  *Quantitative Finance* for an applied version.
