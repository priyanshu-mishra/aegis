"""Model validation report (``docs/MODEL_CARD.md``), structured like an SR 11-7 model-risk review.

Every figure is read from the exported site data (``site/public/data``), so the report can't drift
from the numbers the site and README show. ``tests/test_model_card.py`` checks the committed file
matches a fresh render of the committed data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tabulate import tabulate

from aegis.config import AegisConfig
from aegis.metrics import kupiec_pof, rolling_coverage

# Ongoing-monitoring thresholds. A breach of any one triggers a recalibration review.
COVERAGE_FLOOR_63D = 0.85  # trailing quarter coverage of the 95% interval
KUPIEC_P_FLOOR = 0.05  # 99% VaR breach rate over the trailing year
BASEL_YELLOW = 5  # exceptions in the trailing 250 days


def _pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _p(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _table(rows: list[list], headers: list[str], align: tuple[str, ...]) -> str:
    return tabulate(rows, headers=headers, tablefmt="github", colalign=align, disable_numparse=True)


def _load(data: Path, name: str):
    return json.loads((data / f"{name}.json").read_text())


def current_monitoring(data: Path, alpha99: float = 0.01) -> dict:
    """Each monitoring statistic on the latest data, next to its trigger."""
    vol, var, basel = _load(data, "vol"), _load(data, "var"), _load(data, "basel")
    y, lo, hi = vol["series"]["y"], vol["series"]["aci_lo"], vol["series"]["aci_hi"]
    days = [
        (d, a <= b <= c)
        for d, a, b, c in zip(vol["dates"], lo, y, hi, strict=True)
        if None not in (a, b, c)
    ]
    inside = [x for _, x in days]
    cov63 = rolling_coverage(np.array(inside), 63)
    exceptions = np.array(basel["series"]["aci_exceptions"])  # the Basel count the site shows
    flags = [bool(f) for f in var["0.01"]["series"]["aci_var_breach"] if f is not None][-250:]
    return {
        "as_of": vol["dates"][-1],
        "coverage_63d": float(cov63[-1]),
        "breaches_250d": int(exceptions[-1]),
        "kupiec_p_250d": kupiec_pof(flags, alpha99).p_value,
        # How often each trigger would have fired over the test period, as a share of days.
        "coverage_fired": float(np.mean(cov63[~np.isnan(cov63)] < COVERAGE_FLOOR_63D)),
        "coverage_years": sorted(
            {d[:4] for (d, _), c in zip(days, cov63, strict=True) if c < COVERAGE_FLOOR_63D}
        ),
        "basel_fired": float(np.mean(exceptions >= BASEL_YELLOW)),
        "basel_years": sorted(
            {d[:4] for d, e in zip(basel["dates"], exceptions, strict=True) if e >= BASEL_YELLOW}
        ),
    }


def _pts(x: float, d: int = 1) -> str:
    return f"{100 * x:+.{d}f}"


def _significance(u: dict) -> str:
    """Paired comparisons with their bootstrap intervals; 'established' means the interval excludes 0."""

    def row(label: str, d: dict) -> list[str]:
        ok = d["lo"] > 0 or d["hi"] < 0
        return [
            label,
            f"{_pts(d['estimate'])} points",
            f"{_pts(d['lo'])} to {_pts(d['hi'])}",
            "established" if ok else "**not established**",
        ]

    diff, b01, b05 = u["diff"], u["breach_diff"]["0.01"], u["breach_diff"]["0.05"]
    return _table(
        [
            row("Interval coverage, ACI minus split conformal", diff["aci_minus_split_coverage"]),
            row("Same, 2020 only", diff["aci_minus_split_coverage_2020"]),
            row("Interval coverage, ACI minus Gaussian EWMA", diff["aci_minus_gaussian_coverage"]),
            row("99% VaR breach rate, ACI minus Gaussian EWMA", b01["aci_minus_gaussian"]),
            row(
                "99% VaR breach rate, ACI minus filtered historical simulation",
                b01["aci_minus_fhs"],
            ),
            row("95% VaR breach rate, ACI minus Gaussian EWMA", b05["aci_minus_gaussian"]),
            row(
                "95% VaR breach rate, ACI minus filtered historical simulation",
                b05["aci_minus_fhs"],
            ),
        ],
        ["Comparison", "Difference", "95% interval", "Verdict"],
        ("left", "right", "right", "left"),
    )


def render(data: Path, cfg: AegisConfig) -> str:
    meta, m = _load(data, "meta"), _load(data, "metrics")
    u = _load(data, "uncertainty")
    sens = _load(data, "sensitivity_default")
    abl = _load(data, "ablation_default")
    # SLM sign agreement on near-noise-free synthetic headlines, before and after the schema fix.
    acc1, acc2 = _load(data, "slm_accuracy"), _load(data, "slm_accuracy_v2")
    hi = max(acc1, key=float)
    sign1, sign2 = acc1[hi]["slm"]["sign_agree"], acc2[hi]["slm"]["sign_agree"]
    vm, v99, v95 = m["vol"], m["var"]["0.01"], m["var"]["0.05"]
    es, basel, st = m["es"], m["basel"], m["strategies"]
    sp, level = meta["splits"], 1 - meta["alpha"]
    test_years = f"{sp['test']['start'][:4]}–{sp['test']['end'][:4]}"
    mon = current_monitoring(data)
    c = cfg.conformal

    # Champion (ACI), incumbent (Gaussian EWMA / RiskMetrics) and the strongest classical challenger (FHS).
    def row(label, vol_key, var_key, es_key):
        v, r1, r5, e, b = vm.get(vol_key), v99[var_key], v95[var_key], es[es_key], basel[es_key]
        return [
            label,
            _pct(v["coverage"]) if v else "n/a",
            _pct(v["by_year"]["2020"]) if v else "n/a",
            _pct(v["worst_rolling_coverage"]) if v else "n/a",
            _pct(r5["breach_rate"], 2),
            _pct(r1["breach_rate"], 2),
            _p(r1["kupiec_p"]),
            _p(r1["christoffersen_p"]),
            f"{e['z2']:+.2f}",
            _pct(b["green"]),
            _pct(b["mean_capital"]),
        ]

    champion = _table(
        [
            row("**Champion:** adaptive conformal (ACI)", "aci", "aci_var", "aci"),
            row("Incumbent: Gaussian EWMA (RiskMetrics)", "gaussian", "gaussian_var", "gaussian"),
            row("Challenger: split conformal", "split_absolute", "split_var", "split"),
            row("Challenger: filtered historical simulation", None, "fhs_var", "fhs"),
        ],
        [
            "Model",
            f"{level:.0%} interval coverage",
            "2020 coverage",
            "Worst quarter",
            "95% VaR breaches",
            "99% VaR breaches",
            "Kupiec p (99%)",
            "Christoffersen p (99%)",
            "ES Z2",
            "Basel green",
            "Mean capital",
        ],
        ("left",) + ("right",) * 10,
    )

    regime = _table(
        [
            [
                name,
                *(_pct(vm[k]["by_year"][y]) for y in sorted(vm["aci"]["by_year"])),
                *(_pct(vm[k]["by_tercile"][t]) for t in ("low", "mid", "high")),
            ]
            for name, k in (
                ("ACI", "aci"),
                ("Gaussian EWMA", "gaussian"),
                ("Split conformal", "split_absolute"),
                ("Mondrian (VIX terciles)", "mondrian"),
            )
        ],
        ["Model", *sorted(vm["aci"]["by_year"]), "Calm VIX", "Mid VIX", "Stressed VIX"],
        ("left",) + ("right",) * (len(vm["aci"]["by_year"]) + 3),
    )

    g = sens["grid"]
    s_alpha = _table(
        [
            [
                f"{1 - float(a):.0%}",
                _pct(v["split"]["coverage"]),
                _pct(v["aci"]["coverage"]),
                f"{v['aci']['width']:.3f}",
                _pct(v["aci"]["worst_quarter"]),
            ]
            for a, v in g["alpha"].items()
        ],
        ["Target", "Split coverage", "ACI coverage", "ACI width", "ACI worst quarter"],
        ("left", "right", "right", "right", "right"),
    )
    s_gamma = _table(
        [
            [
                f"{float(k):g}" + (" (default)" if float(k) == c.aci_gamma else ""),
                _pct(v["coverage"]),
                f"{v['width']:.3f}",
                _pct(v["worst_quarter"]),
            ]
            for k, v in g["gamma"].items()
        ],
        ["ACI step γ", "Coverage", "Width", "Worst quarter"],
        ("left", "right", "right", "right"),
    )
    s_window = _table(
        [
            [
                k + (" (default)" if int(k) == c.rolling_window else ""),
                _pct(v["coverage"]),
                f"{v['width']:.3f}",
                _pct(v["worst_quarter"]),
            ]
            for k, v in g["window"].items()
        ],
        ["Rolling window (days)", "Coverage", "Width", "Worst quarter"],
        ("left", "right", "right", "right"),
    )
    lam = sens.get("lambda") or {}
    s_lambda = _table(
        [
            [
                k + (" (default)" if float(k) == cfg.risk.ewma_lambda else ""),
                f"{v['rmse']:.3f}",
                _pct(v["aci_coverage"]),
                f"{v['aci_width']:.3f}",
                _pct(v["gaussian_var99_breach"], 2),
            ]
            for k, v in lam.items()
        ],
        ["EWMA λ", "Point RMSE", "ACI coverage", "ACI width", "Gaussian 99% VaR breaches"],
        ("left", "right", "right", "right", "right"),
    )
    gam_cov = [v["coverage"] for v in g["gamma"].values()]
    lam_cov = [v["aci_coverage"] for v in lam.values()]
    # Largest gap between ACI coverage and its target anywhere in the grid.
    aci_gap = max(
        [abs(v["aci"]["coverage"] - (1 - float(a))) for a, v in g["alpha"].items()]
        + [abs(x - level) for x in gam_cov + lam_cov]
    )
    aci_worst_year = min(vm["aci"]["by_year"], key=vm["aci"]["by_year"].get)

    base = abl[0]["rmse"]
    lex = next(r for r in abl if "lexicon" in r["variant"])
    sizing = _table(
        [
            [
                name,
                f"{st[k]['sharpe']:.2f}",
                _pct(st[k]["ann_vol"]),
                _pct(st[k]["max_drawdown"]),
                _pct(st[k]["overshoot_share"]),
                _pct(st[k]["limit_breach_rate"], 2),
            ]
            for name, k in (
                ("Buy and hold", "buy_hold"),
                ("Vol target, EWMA", "vt_ewma"),
                ("Vol target, point forecast", "vt_point"),
                ("Vol target, conformal upper bound", "vt_conformal"),
            )
        ],
        [
            "Strategy",
            "Sharpe",
            "Volatility",
            "Max drawdown",
            "Days over target",
            "2% loss-limit breaches",
        ],
        ("left", "right", "right", "right", "right", "right"),
    )

    monitor = _table(
        [
            [
                f"Coverage of the {level:.0%} interval, trailing 63 days",
                f"below {_pct(COVERAGE_FLOOR_63D, 0)}",
                _pct(mon["coverage_63d"]),
                "ok" if mon["coverage_63d"] >= COVERAGE_FLOOR_63D else "**review**",
            ],
            [
                "99% VaR exceptions, trailing 250 days (Basel)",
                f"{BASEL_YELLOW} or more (yellow zone)",
                str(mon["breaches_250d"]),
                "ok" if mon["breaches_250d"] < BASEL_YELLOW else "**review**",
            ],
            [
                "Kupiec p, 99% VaR, trailing 250 days",
                f"below {KUPIEC_P_FLOOR}",
                _p(mon["kupiec_p_250d"]),
                "ok" if mon["kupiec_p_250d"] >= KUPIEC_P_FLOOR else "**review**",
            ],
            [
                "Nightly data refresh (`status.json`)",
                "failed",
                "succeeded" if _load(data, "status").get("fresh") else "failed",
                "ok" if _load(data, "status").get("fresh") else "**review**",
            ],
        ],
        ["Statistic", "Trigger", f"Value on {mon['as_of']}", "Status"],
        ("left", "left", "right", "left"),
    )

    return f"""# Model validation report: Aegis conformal risk engine

<!-- Generated by `python main.py model-card` from site/public/data. Do not edit by hand. -->

Structured after the Federal Reserve's SR 11-7 guidance on model risk management: purpose, data,
assumptions, performance against an incumbent, sensitivity, limitations and ongoing monitoring.
This is a self-assessment by the model's developer, not an independent validation, and the model
is a research project, not one used to manage money.

**Data as of {meta["data"]["last_day"]}** · test period {sp["test"]["start"]} to {sp["test"]["end"]}
({sp["test"]["rows"]:,} trading days) · run `{meta["git_sha"]}` · Aegis {meta["aegis_version"]}

## 1. Summary

| | |
|---|---|
| Model | Five-day portfolio volatility forecast with a conformal {level:.0%} interval, plus one-day VaR and Expected Shortfall |
| Champion | Adaptive conformal inference (ACI), γ = {c.aci_gamma:g}, absolute-residual score |
| Incumbent | Gaussian EWMA (RiskMetrics), λ = {cfg.risk.ewma_lambda} |
| Verdict | **Fit for purpose as a calibrated risk forecast.** ACI holds its {level:.0%} promise overall ({_pct(vm["aci"]["coverage"])}) and through 2020 ({_pct(vm["aci"]["by_year"]["2020"])}); its 99% VaR passes Kupiec (p {_p(v99["aci_var"]["kupiec_p"])}) and stays in the Basel green zone on {_pct(basel["aci"]["green"])} of days, against {_pct(basel["gaussian"]["green"])} for the incumbent. |
| Findings against | 99% breaches cluster (Christoffersen p {_p(v99["aci_var"]["christoffersen_p"])}); calibration costs more capital than the incumbent ({_pct(basel["aci"]["mean_capital"])} vs {_pct(basel["gaussian"]["mean_capital"])}); filtered historical simulation is as well calibrated for less ({_pct(basel["fhs"]["mean_capital"])}). |

## 2. Purpose and intended use

The model answers one question each trading day: *how volatile will this portfolio be over the
next {meta["horizon"]} days, and how sure can I be?* It returns a point forecast of realized
volatility, an interval that should contain the outcome {level:.0%} of the time, and one-day VaR
and Expected Shortfall. Intended uses are risk limits, volatility targeting and capital estimates
for a small multi-asset portfolio. It is not intended for single-name risk, intraday risk,
options, or as a return forecast.

## 3. Data

| | |
|---|---|
| Source | {meta["data"]["source"]} daily adjusted closes; VIX as a feature; {meta["data"]["news_items"]} FOMC statements (not used, section 7) |
| Portfolio | Equal weights in {", ".join(meta["data"]["tickers"])}, rebalanced daily |
| History | {meta["data"]["first_day"]} to {meta["data"]["last_day"]} |
| Training | {sp["train"]["start"]} to {sp["train"]["end"]} ({sp["train"]["rows"]:,} days): point model fit |
| Calibration | {sp["cal"]["start"]} to {sp["cal"]["end"]} ({sp["cal"]["rows"]:,} days): conformal scores |
| Test | {sp["test"]["start"]} to {sp["test"]["end"]} ({sp["test"]["rows"]:,} days): never seen in fitting |
| Embargo | {cfg.splits.embargo_days} days between splits, so overlapping {meta["horizon"]}-day targets can't leak |

Prices are free Yahoo Finance data, so dividend and split adjustments are Yahoo's. Look-ahead is
guarded by tests that perturb data after a date and check that nothing known on that date changes.

## 4. Methodology and assumptions

1. **Point model.** {meta["model"]["kind"].capitalize()} regression on HAR-style realized-volatility lags, EWMA volatilities
   and the VIX level and change, predicting log {meta["horizon"]}-day realized volatility (out-of-fold RMSE {meta["model"]["oof_rmse"]:.3f}).
2. **Conformal layer.** Absolute residuals on the calibration set give the initial cut-off. ACI then
   moves its working miss rate by γ after every observed hit or miss, with outcomes arriving
   {meta["horizon"]} days late, and a floor of {c.aci_floor_ratio}·α so it can't run to an infinite interval.
3. **VaR and ES.** One-sided conformal lower bounds on next-day portfolio return. ES uses the average
   calibration score beyond the VaR cut-off.

Key assumption: split conformal assumes calibration and test errors are exchangeable. Markets
violate this (section 5); ACI drops the assumption and instead guarantees long-run average coverage
for any sequence, at the cost of reacting only after misses.

## 5. Performance: champion against incumbent and challengers

{champion}

Test period {test_years}. VaR breach targets are 5% and 1%; Kupiec tests the breach rate,
Christoffersen tests whether breaches cluster (p below 0.05 rejects). ES Z2 is the Acerbi–Székely
statistic (near zero is good, strongly negative means risk understated). Basel columns use a
trailing 250-day window from {basel["aci"]["days"]:,} scored days.

**Stability across regimes.** Coverage of the {level:.0%} interval by year and by VIX level:

{regime}

The textbook split method under-covers in the 2020 crash ({_pct(vm["split_absolute"]["by_year"]["2020"])}); ACI holds {_pct(vm["aci"]["by_year"]["2020"])}.
ACI's weakest year is {aci_worst_year} ({_pct(vm["aci"]["by_year"][aci_worst_year])}). Mondrian conformal (separate cut-offs per VIX tercile) lifts stressed-market
coverage but under-covers calm markets because the tercile boundaries drifted after calibration.

**Is the difference luck?** Breaches and misses come in bursts, so a plain error bar would be too
narrow. These are {u["level"]:.0%} moving-block bootstrap intervals ({u["block_days"]}-day blocks,
{u["draws"]:,} draws), paired so that days that were hard for every method cancel out.
ACI's own coverage is {_pct(u["coverage"]["aci"]["estimate"])} (interval {_pct(u["coverage"]["aci"]["lo"])} to {_pct(u["coverage"]["aci"]["hi"])}); in 2020 alone it is
{_pct(u["coverage_2020"]["aci"]["estimate"])} ({_pct(u["coverage_2020"]["aci"]["lo"])} to {_pct(u["coverage_2020"]["aci"]["hi"])}), wide because a single year is about
250 overlapping days. The interval for overall coverage barely moves with the block length
({_pct(u["aci_coverage_by_block"]["5"]["lo"])} to {_pct(u["aci_coverage_by_block"]["5"]["hi"])} at 5 days, {_pct(u["aci_coverage_by_block"]["63"]["lo"])} to {_pct(u["aci_coverage_by_block"]["63"]["hi"])} at 63).

{_significance(u)}

**In use: volatility targeting at {meta["strategy"]["target_vol"]:.0%}.**

{sizing}

## 6. Sensitivity analysis

Coverage and width as the settings vary, on the saved forecasts (α, γ, window) or with a full
re-run (λ). Width is in log-volatility units.

**Target level α.** ACI tracks every target; split conformal falls short at each one.

{s_alpha}

**ACI step γ.** Coverage stays between {_pct(min(gam_cov))} and {_pct(max(gam_cov))} across a {max(map(float, g["gamma"])) / min(map(float, g["gamma"])):.0f}× range;
a larger step mostly buys wider intervals.

{s_gamma}

**Rolling calibration window.**

{s_window}

**EWMA decay λ.** ACI coverage stays between {_pct(min(lam_cov), 2)} and {_pct(max(lam_cov), 2)}.

{s_lambda}

Across every α, γ and λ tested, ACI coverage is within {100 * aci_gap:.2f} points of its target. The settings
change width, not calibration.

## 7. Limitations and known weaknesses

1. **Clustered tail breaches.** At 99%, ACI's breach rate is on target ({_pct(v99["aci_var"]["breach_rate"], 2)}) but breaches
   bunch (Christoffersen p {_p(v99["aci_var"]["christoffersen_p"])}). It adapts after a miss, so the first days of a shock get through.
2. **Capital cost.** Being calibrated costs capital: mean Basel charge {_pct(basel["aci"]["mean_capital"])} of portfolio value for ACI
   against {_pct(basel["gaussian"]["mean_capital"])} for the incumbent.
3. **A classical model does as well.** Filtered historical simulation passes both tests at 99%
   (Kupiec p {_p(v99["fhs_var"]["kupiec_p"])}, Christoffersen p {_p(v99["fhs_var"]["christoffersen_p"])}) with {_pct(basel["fhs"]["mean_capital"])} capital. Conformal's
   advantage is that it needs no model of the return distribution, not superior calibration.
4. **Text adds nothing on real data.** FOMC statements raised forecast error by {_pct(lex["rmse"] / base - 1)}
   with the lexicon, so the production configuration is price-only.
5. **Narrow evidence.** The 2020 coverage figure has a wide interval ({_pct(u["coverage_2020"]["aci"]["lo"])} to {_pct(u["coverage_2020"]["aci"]["hi"])}), and ACI is statistically
   indistinguishable from filtered historical simulation on breach rate (section 5). One four-asset portfolio, one test period with one crash, free data, flat
   {meta["strategy"]["cost_bps"]:g} bp trading costs. The 10-day regulatory test has only {m["var_long"]["aci"]["n_nonoverlap"]} non-overlapping windows.

**Issues found in development and fixed** (each now has a regression test):

- The SLM schema invited sign errors; direction-named fields took sign agreement from {_pct(sign1, 0)} to {_pct(sign2, 0)}.
- At 99%, ACI's working level could reach zero after a cluster of misses, giving infinite VaR for
  years; the floor in section 4 fixes it.
- Metrics silently dropped infinite-VaR days, flattering the breach rate; they now count as days
  and are reported separately.
- DtACI under-covered with delayed feedback until its experts were scored on the levels actually issued.

## 8. Ongoing monitoring

The data refreshes every weeknight. A breach of any trigger below calls for a recalibration review:
re-check the calibration window, refit the point model, and re-run this report.

{monitor}

Over the test period the coverage trigger would have fired on {_pct(mon["coverage_fired"])} of days (in
{" and ".join(mon["coverage_years"]) or "no year"}) and the Basel trigger on {_pct(mon["basel_fired"])} (in {" and ".join(mon["basel_years"]) or "no year"}): rare enough
not to cry wolf, but not silent in a stress period.

Further triggers that need a person: a change in portfolio composition, a structural market change
(new volatility regime outside the calibration range), or a change to any model setting in section 6.

## 9. Reproduction

```bash
pip install -e .
python main.py fetch -c configs/default.yaml
python main.py backtest -c configs/default.yaml
python main.py sensitivity -c configs/default.yaml
python main.py export-site -c configs/default.yaml
python main.py model-card -c configs/default.yaml
```

Source: [github.com/priyanshu-mishra/aegis](https://github.com/priyanshu-mishra/aegis) ·
guided study: [priyanshu-mishra.github.io/aegis](https://priyanshu-mishra.github.io/aegis/)
"""


def write_model_card(data: Path, cfg: AegisConfig, out: Path) -> Path:
    out.write_text(render(data, cfg))
    return out
