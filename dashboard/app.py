"""Aegis dashboard: browse saved backtest runs.

Run with ``python main.py dashboard`` (or ``streamlit run dashboard/app.py``). Reads the files
that ``backtest`` and ``ablation`` write under ``reports/<run>/``; it never recomputes models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from aegis.backtest import BacktestResult
from aegis.metrics import is_covered, rolling_coverage
from aegis.report import LABELS, ROLL, compute_metrics

REPORTS = Path(os.environ.get("AEGIS_REPORTS", Path(__file__).resolve().parents[1] / "reports"))
# Reference palette: categorical slots in fixed order, plus chrome and status.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED, CRITICAL = "#898781", "#d03b3b"
DATE_AXIS = alt.Axis(format="%Y", tickCount="year", labelAngle=0)

st.set_page_config(page_title="Aegis risk engine", layout="wide")


@st.cache_data
def load_run(path: str) -> tuple[BacktestResult, dict]:
    res = BacktestResult.load(Path(path))
    return res, compute_metrics(res)


runs = sorted(p.parent.name for p in REPORTS.glob("*/run_info.json"))
if not runs:
    st.title("Aegis")
    st.info("No saved runs yet. Run `python main.py demo` or `python main.py backtest`.")
    st.stop()

run = st.sidebar.selectbox("Run", runs, index=runs.index("default") if "default" in runs else 0)
res, metrics = load_run(str(REPORTS / run))
info, alpha = res.info, res.info["alpha"]
vm = metrics["vol"]
methods = info["vol_methods"]

st.title(f"Aegis · {run}")
sp, d = info["splits"], info["data"]
st.caption(
    f"{d['source']} · {', '.join(d['tickers'])} · test {sp['test']['start']} → "
    f"{sp['test']['end']} ({sp['test']['rows']} days) · point model {info['model']['kind']} · "
    f"α = {alpha:g}"
)

# ---- Headline tiles ----------------------------------------------------------------------
best = min(methods, key=lambda m: abs(vm[m]["coverage"] - (1 - alpha)))
cols = st.columns(4)
cols[0].metric("ACI coverage", f"{vm['aci']['coverage']:.1%}" if "aci" in vm else "–",
               help=f"Target {1 - alpha:.0%}, 5-day realized vol intervals")  # fmt: skip
cols[1].metric("ACI worst quarter", f"{vm['aci']['worst_rolling_coverage']:.1%}" if "aci" in vm
               else "–")  # fmt: skip
if res.var:
    a99 = min(res.var)
    v = metrics["var"][str(a99)].get("aci_var")
    cols[2].metric(f"{(1 - a99) * 100:g}% ACI VaR breaches",
                   f"{v['breach_rate']:.2%}" if v else "–", help=f"Target {a99:.0%}")  # fmt: skip
cols[3].metric("Closest to target", f"{vm[best]['coverage']:.1%}",
               help=LABELS.get(best, best))  # fmt: skip
cols[3].caption(LABELS.get(best, best))

# Only the selected view is rendered (dataframes in hidden st.tabs render at zero width).
view = (
    st.segmented_control(
        "View",
        ["Intervals", "Coverage", "VaR", "Text ablation", "Latency"],
        default="Intervals",
        label_visibility="collapsed",
    )
    or "Intervals"
)

# ---- Intervals ---------------------------------------------------------------------------
if view == "Intervals":
    m = st.selectbox("Method", methods, index=methods.index("aci") if "aci" in methods else 0,
                     format_func=lambda k: LABELS.get(k, k))  # fmt: skip
    vol = res.vol
    to_ann = np.sqrt(252) * 100
    df = pd.DataFrame({
        "date": vol.index,
        "realized": np.exp(vol["y"]) * to_ann,
        "lo": np.exp(np.clip(vol[f"{m}_lo"], -20, 5)) * to_ann,
        "hi": np.exp(np.clip(vol[f"{m}_hi"], -20, 5)) * to_ann,
    })  # fmt: skip
    df["miss"] = ~np.asarray(is_covered(vol["y"], vol[f"{m}_lo"], vol[f"{m}_hi"]))
    y_dom = [df["realized"].min() / 2, df["realized"].max() * 2]
    scale = alt.Scale(type="log", domain=y_dom, clamp=True)
    base = alt.Chart(df).encode(x=alt.X("date:T", title=None, axis=DATE_AXIS))
    band = base.mark_area(opacity=0.25, color=SERIES[0]).encode(
        y=alt.Y("lo:Q", scale=scale, title="annualized vol, %"), y2="hi:Q"
    )
    line = base.mark_line(color=MUTED, strokeWidth=1).encode(
        y=alt.Y("realized:Q", scale=scale),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("realized:Q", format=".1f"),
                 alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f")],
    )  # fmt: skip
    misses = (
        base.transform_filter("datum.miss")
        .mark_circle(size=40, color=CRITICAL)
        .encode(y=alt.Y("realized:Q", scale=scale))
    )
    st.altair_chart((band + line + misses).properties(height=380).interactive(bind_y=False),
                    width="stretch")  # fmt: skip
    st.caption(f"Shaded: {1 - alpha:.0%} interval. Red: realized vol outside it "
               f"({df['miss'].mean():.1%} of days).")  # fmt: skip

# ---- Coverage ----------------------------------------------------------------------------
if view == "Coverage":
    rows = []
    for k, v in vm.items():
        t = v["by_tercile"]
        rows.append({
            "Method": LABELS.get(k, k), "Coverage": v["coverage"], "Avg width": v["mean_width"],
            "∞ days": v["infinite_days"], "Winkler": v["winkler"],
            f"Worst {ROLL}d": v["worst_rolling_coverage"], "Low vol": t.get("low"),
            "Mid vol": t.get("mid"), "High vol": t.get("high"),
        })  # fmt: skip
    table = pd.DataFrame(rows).set_index("Method")
    pct = ["Coverage", f"Worst {ROLL}d", "Low vol", "Mid vol", "High vol"]
    st.dataframe(table.style.format({c: "{:.1%}" for c in pct} | {"Avg width": "{:.3f}",
                                                                   "Winkler": "{:.3f}"}),
                 width="stretch")  # fmt: skip

    pick = st.multiselect("Rolling coverage", methods,
                          default=[x for x in ("split_absolute", "cqr", "aci") if x in methods][:3],
                          max_selections=4, format_func=lambda k: LABELS.get(k, k))  # fmt: skip
    if pick:
        roll = pd.DataFrame({"date": res.vol.index})
        for k in pick:
            cov = np.asarray(is_covered(res.vol["y"], res.vol[f"{k}_lo"], res.vol[f"{k}_hi"]))
            roll[LABELS.get(k, k)] = rolling_coverage(cov, ROLL)
        long = roll.melt("date", var_name="method", value_name="coverage").dropna()
        order = [LABELS.get(k, k) for k in pick]  # color follows the method, not its rank
        lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None, axis=DATE_AXIS),
            y=alt.Y("coverage:Q", title=f"{ROLL}-day coverage", scale=alt.Scale(zero=False),
                    axis=alt.Axis(format="%")),
            color=alt.Color("method:N", scale=alt.Scale(domain=order, range=SERIES[: len(order)]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["date:T", "method:N", alt.Tooltip("coverage:Q", format=".1%")],
        )  # fmt: skip
        target = (
            alt.Chart(pd.DataFrame({"t": [1 - alpha]}))
            .mark_rule(color=MUTED, strokeDash=[4, 4])
            .encode(y=alt.Y("t:Q", title=None))
        )
        st.altair_chart((lines + target).properties(height=320), width="stretch")

    years = sorted({y for v in vm.values() for y in v["by_year"]})
    by_year = pd.DataFrame(
        {LABELS.get(k, k): [v["by_year"].get(y) for y in years] for k, v in vm.items()},
        index=[str(y) for y in years],
    ).T
    st.markdown("**Coverage by year**")
    st.dataframe(by_year.style.format("{:.1%}"), width="stretch")

# ---- VaR ---------------------------------------------------------------------------------
if view == "VaR":
    if not res.var:
        st.info("This run has no VaR backtest.")
    else:
        a = st.radio("Level", sorted(res.var, reverse=True), horizontal=True,
                     format_func=lambda x: f"{(1 - x) * 100:g}% VaR")  # fmt: skip
        mv = metrics["var"][str(a)]
        vt = pd.DataFrame([{
            "Method": LABELS.get(k, k), "Breaches": v["breach_rate"], "Kupiec p": v["kupiec_p"],
            "Christoffersen p": v["christoffersen_p"], "Conditional p": v["conditional_p"],
            "Avg VaR": v["mean_var"],
        } for k, v in mv.items()]).set_index("Method")  # fmt: skip

        def _pass(p):
            return "color: #0ca30c" if p > 0.05 else f"color: {CRITICAL}"

        st.dataframe(
            vt.style.format({"Breaches": "{:.2%}", "Avg VaR": "{:.2%}", "Kupiec p": "{:.3f}",
                             "Christoffersen p": "{:.3f}", "Conditional p": "{:.3f}"})
            .map(_pass, subset=["Kupiec p", "Christoffersen p", "Conditional p"]),
            width="stretch",
        )  # fmt: skip
        st.caption(f"Target breach rate {a:.0%}. p > 0.05 (green ✓) means the test does not "
                   "reject; red means it does.")  # fmt: skip
        vdf = res.var[a]
        choose = [c for c in ("aci_var", "gaussian_var") if c in vdf]
        plot = pd.DataFrame({"date": vdf.index, "return": vdf["r_next"] * 100})
        for c in choose:
            plot[LABELS[c]] = vdf[c] * 100
        long = plot.melt(["date", "return"], var_name="model", value_name="VaR")
        order = [LABELS[c] for c in choose]
        ret = (
            alt.Chart(plot)
            .mark_line(color=MUTED, strokeWidth=0.7, opacity=0.6)
            .encode(
                x=alt.X("date:T", title=None, axis=DATE_AXIS),
                y=alt.Y("return:Q", title="next-day return, %"),
            )
        )
        var_lines = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
            x="date:T", y="VaR:Q",
            color=alt.Color("model:N", scale=alt.Scale(domain=order, range=SERIES[:len(order)]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["date:T", "model:N", alt.Tooltip("VaR:Q", format=".2f")],
        )  # fmt: skip
        st.altair_chart((ret + var_lines).properties(height=360).interactive(bind_y=False),
                        width="stretch")  # fmt: skip

# ---- Risk measures (ES, 10-day VaR, Basel, decomposition) --------------------------------
if view == "Risk measures":
    from aegis.report import RISK_LABELS

    if "es" not in metrics:
        st.info("This run predates the risk-measure extensions; re-run `backtest`.")
    else:
        rm = info["risk_measures"]
        st.markdown(f"**Expected Shortfall {(1 - rm['es_alpha']) * 100:g}%**: Acerbi–Székely Z2 "
                    "is 0 when ES is right, negative when tail losses exceed it.")  # fmt: skip
        es_t = pd.DataFrame([{"Method": RISK_LABELS[k], "Breaches": v["breach_rate"],
                              "Avg ES": v["mean_es"], "Z2": v["z2"], "Zone": v["zone"]}
                             for k, v in metrics["es"].items()]).set_index("Method")  # fmt: skip
        st.dataframe(es_t.style.format({"Breaches": "{:.2%}", "Avg ES": "{:.2%}",
                                        "Z2": "{:+.2f}"}), width="stretch")  # fmt: skip
        st.markdown(f"**{rm['long_horizon']}-day {(1 - rm['long_alpha']) * 100:g}% VaR** "
                    "(tests on non-overlapping windows)")  # fmt: skip
        lv = pd.DataFrame([{"Method": RISK_LABELS[k], "Breaches": v["breach_rate_nonoverlap"],
                            "Kupiec p": v["kupiec_p"], "Avg VaR": v["mean_var"]}
                           for k, v in metrics["var_long"].items()])  # fmt: skip
        lv = lv.set_index("Method")
        st.dataframe(lv.style.format({"Breaches": "{:.2%}", "Kupiec p": "{:.3f}",
                                      "Avg VaR": "{:.2%}"}), width="stretch")  # fmt: skip
        if "basel" in metrics and res.basel is not None:
            st.markdown("**Basel traffic light**: 99% VaR exceptions in the last 250 days")
            bt = pd.DataFrame([{"Method": RISK_LABELS[k], "Days green": v["green"],
                                "Max exceptions": v["max_exceptions"],
                                "Avg multiplier": v["mean_multiplier"],
                                "Avg capital": v["mean_capital"]}
                               for k, v in metrics["basel"].items()])  # fmt: skip
            bt = bt.set_index("Method")
            st.dataframe(bt.style.format({"Days green": "{:.0%}", "Avg multiplier": "{:.2f}",
                                          "Avg capital": "{:.1%}"}), width="stretch")  # fmt: skip
            pick = ["aci", "gaussian", "hist"]
            ex = res.basel[[f"{k}_exceptions" for k in pick]].copy()
            ex.columns = [RISK_LABELS[k] for k in pick]
            long = ex.reset_index(names="date").melt("date", var_name="method",
                                                      value_name="exceptions").dropna()  # fmt: skip
            zones = pd.DataFrame({"lo": [0, 4.5, 9.5], "hi": [4.5, 9.5, 14],
                                  "zone": ["green", "yellow", "red"]})  # fmt: skip
            bands = alt.Chart(zones).mark_rect(opacity=0.1).encode(
                y="lo:Q", y2="hi:Q",
                color=alt.Color("zone:N", scale=alt.Scale(domain=["green", "yellow", "red"],
                                range=["#0ca30c", "#fab219", CRITICAL]), legend=None),
            )  # fmt: skip
            lines = alt.Chart(long).mark_line(strokeWidth=1.8).encode(
                x=alt.X("date:T", title=None, axis=DATE_AXIS),
                y=alt.Y("exceptions:Q", title="exceptions (250d)"),
                color=alt.Color("method:N", scale=alt.Scale(domain=ex.columns.tolist(),
                                range=SERIES[:3]), legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["date:T", "method:N", "exceptions:Q"],
            )  # fmt: skip
            st.altair_chart((bands + lines).properties(height=300), width="stretch")
        if res.contrib is not None:
            st.markdown("**Share of portfolio risk by asset** (Euler; negative = hedge)")
            cl = res.contrib.reset_index(names="date").melt("date", var_name="asset",
                                                            value_name="share")  # fmt: skip
            st.altair_chart(alt.Chart(cl).mark_line(strokeWidth=1.6).encode(
                x=alt.X("date:T", title=None, axis=DATE_AXIS),
                y=alt.Y("share:Q", axis=alt.Axis(format="%"), title=None),
                color=alt.Color("asset:N", scale=alt.Scale(range=SERIES),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["date:T", "asset:N", alt.Tooltip("share:Q", format=".1%")],
            ).properties(height=280), width="stretch")  # fmt: skip

# ---- Ablation ----------------------------------------------------------------------------
if view == "Text ablation":
    path = REPORTS / run / "ablation.json"
    if not path.exists():
        st.info(f"No ablation for this run. Run `python main.py ablation -c configs/{run}.yaml`.")
    else:
        abl = pd.DataFrame(json.loads(path.read_text())).set_index("variant")
        abl.insert(1, "vs prices", abl["rmse"] / abl["rmse"].iloc[0] - 1)
        show = {"rmse": "Vol RMSE", "vs prices": "vs prices", "cqr_coverage": "CQR cov",
                "cqr_width": "CQR width", "aci_coverage": "ACI cov", "aci_width": "ACI width",
                "shock_recovery": "Shock recovery"}  # fmt: skip
        table_abl = abl[[c for c in show if c in abl]].rename(columns=show)
        fmt = {"Vol RMSE": "{:.3f}", "vs prices": "{:+.1%}", "CQR cov": "{:.1%}",
               "CQR width": "{:.3f}", "ACI cov": "{:.1%}", "ACI width": "{:.3f}",
               "Shock recovery": "{:.2f}"}  # fmt: skip
        fmt = {k: v for k, v in fmt.items() if k in table_abl.columns}
        st.dataframe(table_abl.style.format(fmt, na_rep="–"), width="stretch")
        st.caption("Negative 'vs prices' means text improved the volatility forecast.")

# ---- Latency -----------------------------------------------------------------------------
if view == "Latency":
    lat = pd.DataFrame(info["latency"]).T[["calls", "total_s", "p50_ms", "p95_ms"]]
    st.dataframe(lat.style.format({"calls": "{:.0f}", "total_s": "{:.2f}", "p50_ms": "{:.2f}",
                                   "p95_ms": "{:.2f}"}), width="stretch")  # fmt: skip
    if info.get("text", {}).get("used"):
        st.json(info["text"])
