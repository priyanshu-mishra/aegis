# Extending Aegis with AI: what changes, what to build, and what could go wrong

This started as a proposal; **A and B have now been built and tested, and both came out negative**
(results in each section below, and in [POINT_MODELS.md](POINT_MODELS.md) and
[EARLY_WARNING.md](EARLY_WARNING.md)). Each idea is tied to a weakness Aegis has already measured,
so every extension has a hypothesis that the existing backtest can confirm or kill. C to F are still
proposals.

## Where Aegis stands

Aegis already uses AI in two places: a local 3B-parameter language model that reads Fed statements
(prompted and schema-constrained, **not fine-tuned**), and a LightGBM option for the point forecast. The
evidence so far is a caution against assuming "more AI" helps:

| Finding | Number | What it implies |
|---|---|---|
| A flexible learner lost to a simple one | log-vol RMSE 2020–2026: ridge **0.428**, HAR **0.446**, neural net **0.456**, LightGBM **0.471** | With about 2,000 noisy daily observations, extra capacity overfits (extension A, below). |
| Text added nothing on real data | forecast **1.2%** worse (rule-based), **2.8%** worse (language model) | The limit is the data (about 8 statements a year), not the reader. |
| Tail breaches cluster | Christoffersen p = 0.008 at 99% for ACI | Adaptation is reactive: the first days of a shock get through. |
| Calibration costs capital | 30.6% vs 18.0% for RiskMetrics; FHS matched at 19.3% | A cheaper, tighter interval would be worth real money. |
| A narrow evidence base | one portfolio, one crash | Any claim needs a second market and a second crisis. |

## How AI changes finance, and why a calibration layer matters more as it does

1. **Forecasting gets more flexible.** Trees, neural networks and pretrained time-series models can
   capture nonlinear structure that GARCH-style models cannot. The academic evidence (for example Gu,
   Kelly and Xiu, 2020, on asset pricing) is that machine learning does add value, but financial data
   is low signal and shifts over time, so gains are real mainly where data is plentiful. The ridge
   result above is the small-sample version of the same lesson.
2. **Text becomes a usable input.** Language models turn filings, earnings calls, news and central-bank
   communication into numbers cheaply and at scale. The new hazards are *look-ahead bias* (a model
   pretrained on the internet may already "know" what happened after the date it is asked about),
   cost, and run-to-run reproducibility.
3. **Model risk gets harder.** Supervisors apply the same model-risk expectations (in the US, SR 11-7:
   validation, challenger models, ongoing monitoring) to machine-learning models, which are harder to
   explain and validate. That makes **model-agnostic** checks more valuable: a coverage and breach audit
   works the same whether the forecaster is a regression or a neural network.
4. **Operations get automated.** Drafting documentation, monitoring, reconciliation and first-line
   triage are natural targets, with a human signing off.
5. **A systemic question to watch, not a settled fact.** If many firms use the same pretrained models
   or similar signals, their errors may correlate. This is an open research topic and I would present it
   as a risk to study.

**The design principle that follows: AI proposes, statistics certifies.** Conformal prediction is a
wrapper that works around *any* forecaster. That is why it suits this project: you can swap in a more
powerful model and keep the same audit, and the coverage guarantee stays in force whether or not the
new model is good. A bad model gets wider intervals, not hidden failure.

## The candidates

| # | Extension | Issue it targets | Hypothesis | Effort |
|---|---|---|---|---|
| A | Deep or pretrained time-series model as the point forecaster | Interval width | Better forecasts give narrower intervals at the same coverage | Medium |
| B | Learned early-warning of stress to drive adaptation | Clustered breaches | Widening *before* the shock lowers first-days misses | Medium |
| C | Higher-frequency text with LLM or embedding features | Text added nothing | Frequency, not method, was the limit | Medium to high |
| D | Fine-tune a small model on the synthetic world | Latency, prompt fragility | A 0.5B–1.5B model matches the 3B prompted one, much faster | Medium |
| E | Conformal sets around the language model's own outputs | Trusting parsed signals | Abstaining on uncertain parses improves the signal | Low to medium |
| F | LLM-drafted validation and monitoring commentary | Model-risk workload | The report updates itself with every figure checked | Low |

### A. A learned point forecaster inside the same conformal layer

*What.* Replace the ridge regression with a small recurrent or temporal-convolution network on
realised-volatility features, or a pretrained time-series foundation model (Chronos, TimesFM and Moirai
are public examples) used zero-shot. Conformal and baselines stay unchanged.

*Why it is interesting.* Interval width is set by the point model's error, so a better forecaster is
the most direct route to a tighter interval. Because the guarantee is model-agnostic, this is also a
clean test of the "AI proposes, statistics certifies" idea.

*How to test.* Add it as a `point_model` option, run the same walk-forward, and compare **width at
equal coverage** (Winkler score) with a Diebold–Mariano-style test. Expect it to be hard: LightGBM
already lost.

*Hazard.* Pretrained time-series models may have seen SPY, TLT and the 2020 crash during pretraining.
That contaminates the test the same way an LLM does. Prefer models trained on data that ends before 2020,
or treat the result as an upper bound.

**Result (built, `risk.point_model: mlp`).** I added a small neural network as a fourth point model:
five networks averaged, one hidden layer of 16 tanh units, an L2 penalty, standardised inputs. I fixed
that architecture before running it and did not tune it on the test period. Same walk-forward, same
conformal layer; only the point model changes. Pretrained foundation models were not tried (see the
hazard above, and they need PyTorch and a weights download).

| Point model | RMSE | Diebold–Mariano p vs ridge | ACI coverage | ACI width | ACI Winkler |
|---|---|---|---|---|---|
| Ridge (production) | 0.428 | - | 95.0% | 1.738 | 2.211 |
| HAR regression | 0.446 | 0.002 | 95.3% | 1.874 | 2.335 |
| Neural net | 0.456 | <0.001 | 94.6% | 1.831 | 2.318 |
| LightGBM | 0.471 | <0.001 | 94.9% | 2.057 | 2.454 |

Two things hold up. **The conformal layer did its job whichever model sat underneath**: coverage
stayed between 94.6% and 95.3% for all four. And **interval width follows point error**: for split
conformal, width is in exactly the same order as RMSE (ridge, HAR, neural net, LightGBM), and for ACI
nearly so (the neural net and HAR swap). That is the mechanism the idea rested on: a better forecaster
would buy tighter intervals. But the better forecaster did not exist here. Ridge is
significantly best, so the hypothesis "a learned model tightens the intervals" is rejected on this
data. With about 2,000 noisy training days, the simple model wins; the guarantee, not the model,
carries the result. Full table: [POINT_MODELS.md](POINT_MODELS.md).

### B. A learned early-warning signal for stress

*What.* A classifier on cross-asset signals (VIX term structure, rolling correlations, credit spreads
if available) that estimates the probability of a stress regime over the next five days. Feed it to
the Mondrian (regime-conditional) conformal layer or to a schedule for ACI's step size.

*Why.* ACI learns only after a miss. An anticipatory signal moves some of that learning earlier. The
long-run coverage guarantee stays, because ACI remains the backstop.

*How to test.* Does Christoffersen pass at 99%? And, more directly, what is the breach rate in the first
five days of each stress episode (February to March 2020, 2022, 2025) compared with ACI alone? An
event-study table answers it.

*Hazard.* Early-warning classifiers are notorious for false alarms and for fitting the last crisis.
There are only a handful of independent stress episodes, so the evidence will be thin. State that.

**Result (built, `conformal.early_warning: true`).** A logistic regression, fitted on the training
years only, predicts from eight as-of-safe inputs whether any of the next five days breaches a
Gaussian 99% VaR. It has modest skill: **AUC 0.68 on the calibration years, 0.62 on the test years**.
Days are sorted into three regimes by that probability and a new `RegimeACI` runs ACI separately in
each. A control sorts by current volatility instead, so the learned signal has to beat a plain label.

| 99% VaR | Breach rate | Christoffersen p | Breaches in the first 5 days of 15 episodes | Mean VaR |
|---|---|---|---|---|
| ACI, one regime | 0.93% | 0.008 | 10 | 3.14% |
| ACI by current volatility (control) | 0.99% | 0.010 | 10 | 2.71% |
| ACI by learned stress probability | 0.93% | 0.008 | 9 | 2.82% |

The problem is real: 10 breaches in the 75 days that start a stress episode is about 13%, against a
1% target. But **the learned signal did not fix it**. Clustering is unchanged (Christoffersen 0.008),
episode breaches fall from 10 to 9, and at 95% the learned version clusters *more* than one regime
(Christoffersen 0.019 against 0.072). Fifteen episodes is thin evidence, so one breach is noise, but
nothing here supports the claim. A signal with AUC 0.62 simply cannot see the shock early enough.

One useful side finding, which comes from the regime split and not from the learning: regime-conditional
ACI holds the breach rate while shrinking the average 99% VaR by 10 to 14%, and the plain volatility
control does at least as well as the learned signal. Full tables: [EARLY_WARNING.md](EARLY_WARNING.md).

### C. Text that arrives often enough to matter

*What.* The negative result says eight statements a year is too sparse, not that text is useless. Try
sources that arrive more often: Fed speeches and minutes (public), and headlines (usually licensed).
Compare the rule-based reader, the prompted language model and sentence embeddings fed to the
regression, always against the price-only model.

*How to test.* The synthetic world already provides a positive control: if the pipeline cannot recover a
planted signal, the real-data result means nothing. Then the real-data ablation is the same one as today.

*Hazard, and the most important one in this list: look-ahead bias in language models.* A model trained on
data past the test date can recall outcomes. Entity and date masking, which Aegis already does, helps but
is not proof. Stronger checks: ask the model for the date or the outcome and see whether it can answer,
use a model whose training cutoff predates the test window where one exists, and report results only on
text published after the cutoff. Published studies of this problem (for example Glasserman and Lin, 2023)
found that apparent predictive power can come from it. Please verify the citations before you quote them.

### D. Fine-tuning, done where it is justified

*What.* Today's model is prompted, not trained, because there are only 139 unlabeled statements.
But the project's synthetic world generates **unlimited headlines with known shocks**: a labelled
dataset. Fine-tune a small open model (0.5B to 1.5B parameters) with a low-rank adapter (LoRA) on thousands
of synthetic examples, then evaluate on held-out synthetic text and on real FOMC statements.

*Why.* A smaller specialised model should be much faster than 1.8 seconds per text, produce the schema
without prompt tricks, and be deterministic. It also gives the project a genuine fine-tuning story,
built on a sound data source.

*How to test.* Sign agreement and correlation with the true shock on synthetic data (today: 100% and
0.95 for the 3B prompted model), latency, and the real-data ablation unchanged.

*Hazard.* A sim-to-real gap: a model that learns the synthetic templates may not read real central-bank
prose. The real FOMC check is what decides it, and its ceiling is the diversity of the synthetic text.

### E. Calibrating the language model itself

*What.* Use the same conformal machinery on the parser. Produce a *set* of plausible directions for each
shock (up, down, neutral) with a guaranteed coverage level, and let the downstream model down-weight or
ignore parses whose set is not a single answer.

*Why.* It applies the project's central idea one layer up: honest uncertainty about the inputs, not just the
outputs. It is also cheap, since the calibration data is the labelled synthetic set.

### F. A self-updating validation report

*What.* A job that reads the nightly data, evaluates the monitoring triggers in the validation report,
and drafts the commentary ("coverage trigger fired in 2025; breaches clustered around...") with a
language model. A check requires that **every number in the prose equals the number in the data**, in
the same spirit as the existing staleness test.

*Why.* Documentation is a large share of real model-risk work. A drafted, verified, human-approved report
is a realistic, visible use of AI in finance, and the guard against invented figures is built in.

## What I would build next

A and B are done (negative). **D** is the best story if the goal is to show fine-tuning skills, and
**F** is the fastest, lowest-risk addition. **C** is the most interesting research question but needs a
data source and the most care.

Both bets on A and B came in negative, which is itself informative: on about 2,000 daily observations
neither a flexible model nor a learned stress signal beat the simple version. That makes the remaining
ideas more interesting, because they add *information* (C: more text) or *speed and control* (D, E, F)
instead of more model capacity. D should work on synthetic data, and the real-data transfer is the open
question.

## Questions to narrow this down

1. Is the goal **research credibility** (A, B, C), a **fine-tuning demonstration** (D, E) or a **product
   feel** (F)?
2. Are you happy to add optional heavy dependencies (PyTorch, model weights)? The old `[llm]` extra was
   removed because nothing used it; A and D would bring one back, kept optional.
3. Do you have access to a licensed headline or transcript source, or should C stay on public Fed text?
4. Any hard limit on compute (laptop only, or a GPU available for D)?

Verify any paper cited above before putting it in a document with your name on it.
