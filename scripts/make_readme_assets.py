"""Render the README figures (light + dark variants) from saved runs.

Prerequisites (each writes into reports/):

    python main.py backtest -c configs/default.yaml
    python main.py ablation -c configs/default.yaml
    python main.py ablation -c configs/synthetic.yaml
    python main.py ablation -c configs/ollama.yaml      # optional, needs Ollama
    python main.py bench

Then:  python scripts/make_readme_assets.py   →  docs/assets/*.svg
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from aegis.backtest import BacktestResult
from aegis.metrics import is_covered, rolling_coverage
from aegis.report import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "docs" / "assets"

THEMES = {
    "light": {
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "neg": "#e34948", "mid": "#f0efec", "good": "#006300", "bad": "#d03b3b",
        "band": "#2a78d6",
    },
    "dark": {
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "neg": "#e66767", "mid": "#383835", "good": "#0ca30c", "bad": "#e66767",
        "band": "#3987e5",
    },
}  # fmt: skip

plt.rcParams.update({
    "svg.fonttype": "path",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
})  # fmt: skip


def style(ax, t, title=None, subtitle=None, ylabel=None, grid="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    if grid:
        ax.grid(axis=grid, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
    if title:
        ax.set_title(title, loc="left", color=t["ink"], fontsize=13, fontweight="bold", pad=24)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=9.5,
                va="bottom")  # fmt: skip


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


LABEL = {
    "gaussian": "Gaussian (uncalibrated)", "raw_quantile": "Raw quantile model",
    "split_absolute": "Split conformal", "split_normalized": "Split, normalized",
    "cqr": "CQR", "rolling": "Rolling conformal", "aci": "Adaptive conformal (ACI)",
    "dtaci": "DtACI (self-tuning ACI)", "mondrian": "Mondrian (per vol regime)",
}  # fmt: skip


def coverage_dots(res, metrics, t, theme):
    """Dot plot: overall coverage and worst quarter per method, with the 95% target."""
    vm = metrics["vol"]
    order = ["raw_quantile", "split_normalized", "split_absolute", "gaussian", "cqr", "dtaci",
             "rolling", "mondrian", "aci"]  # fmt: skip
    order = [m for m in order if m in vm]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    target = 1 - res.info["alpha"]
    ax.axvline(target, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
    ax.text(target, len(order) - 0.35, "target 95%", color=t["ink2"], fontsize=8.5, ha="center")
    for i, m in enumerate(order):
        v = vm[m]
        hero = m == "aci"
        c = t["series"][0] if hero else t["muted"]
        ax.plot([v["worst_rolling_coverage"], v["coverage"]], [i, i], color=t["grid"], lw=2,
                solid_capstyle="round", zorder=1)  # fmt: skip
        ax.scatter(v["worst_rolling_coverage"], i, s=46, facecolor="none", edgecolor=c, lw=1.6,
                   zorder=2)  # fmt: skip
        ax.scatter(v["coverage"], i, s=70 if hero else 54, color=c, zorder=3)
        ax.text(v["coverage"] + 0.006, i, f"{v['coverage']:.1%}", va="center",
                color=t["ink"] if hero else t["ink2"], fontsize=9,
                fontweight="bold" if hero else "normal")  # fmt: skip
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    for lbl, m in zip(ax.get_yticklabels(), order, strict=True):
        lbl.set_color(t["ink"] if m == "aci" else t["ink2"])
    ax.set_xlim(0.55, 1.0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    style(ax, t, "Coverage of 95% volatility intervals, 2020–2026", grid="x")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", color=t["muted"], markersize=7, label="overall coverage"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=t["muted"],
               markersize=7, mew=1.6, label="worst rolling quarter (63 trading days)"),
    ]  # fmt: skip
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
                    frameon=False, fontsize=9, handletextpad=0.3, borderaxespad=0.2)  # fmt: skip
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    save(fig, "coverage", theme)


def rolling_story(res, t, theme):
    """Rolling coverage: split conformal vs ACI, COVID shaded."""
    vol = res.vol
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.axvspan(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-30"), color=t["muted"],
               alpha=0.12, lw=0)  # fmt: skip
    ax.text(pd.Timestamp("2020-04-10"), 0.632, "COVID\ncrash", ha="center", color=t["ink2"],
            fontsize=8.5)  # fmt: skip
    for i, (m, name) in enumerate([("split_absolute", "Split conformal"), ("aci", "ACI")]):
        cov = np.asarray(is_covered(vol["y"], vol[f"{m}_lo"], vol[f"{m}_hi"]))
        roll = rolling_coverage(cov, 63)
        color = t["muted"] if i == 0 else t["series"][0]
        ax.plot(vol.index, roll, color=color, lw=1.4 if i == 0 else 2.0, label=name, zorder=2 + i)
    ax.axhline(0.95, color=t["muted"], linestyle=(0, (4, 3)), lw=1)
    ax.text(vol.index[-1], 0.953, "target 95%", color=t["ink2"], fontsize=8.5, ha="right",
            va="bottom")  # fmt: skip
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.set_ylim(0.6, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    style(ax, t, "Rolling 63-day coverage: split conformal vs adaptive conformal",
          "Markets are not exchangeable, so a fixed calibration set under-covers after regime "
          "shifts; ACI adapts.")  # fmt: skip
    save(fig, "rolling", theme)


def year_heatmap(metrics, t, theme):
    vm = metrics["vol"]
    order = ["gaussian", "split_absolute", "split_normalized", "cqr", "rolling", "mondrian",
             "dtaci", "aci"]  # fmt: skip
    order = [m for m in order if m in vm]
    years = sorted({int(y) for v in vm.values() for y in v["by_year"]})
    data = np.array([[vm[m]["by_year"].get(y, vm[m]["by_year"].get(str(y))) for y in years]
                     for m in order])  # fmt: skip
    cmap = LinearSegmentedColormap.from_list("div", [t["neg"], t["mid"], t["series"][0]])
    fig, ax = plt.subplots(figsize=(8.6, 0.45 * len(order) + 0.9))
    dev = data - 0.95
    ax.imshow(dev, cmap=cmap, vmin=-0.08, vmax=0.08, aspect="auto")
    for i in range(len(order)):
        for j in range(len(years)):
            v = data[i, j]
            strong = abs(v - 0.95) > 0.065
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if strong else t["ink"])  # fmt: skip
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=t["ink2"], length=0, labelsize=9)
    ax.set_xticks(np.arange(-0.5, len(years)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color="none")
    ax.tick_params(which="minor", length=0)
    for x in np.arange(-0.5, len(years)):
        ax.axvline(x, color=t["mid"] if theme == "dark" else "#ffffff", lw=2)
    for y in np.arange(-0.5, len(order)):
        ax.axhline(y, color=t["mid"] if theme == "dark" else "#ffffff", lw=2)
    ax.set_title("Coverage by year (target 95%)", loc="left", color=t["ink"], fontsize=13,
                 fontweight="bold", pad=24)  # fmt: skip
    ax.text(0, 1.03, "Red: below target · blue: above target · gray: on target",
            transform=ax.transAxes, color=t["ink2"], fontsize=9.5)  # fmt: skip
    save(fig, "by_year", theme)


def var_chart(metrics, t, theme):
    """99% VaR breach rate per method vs the 1% target, with test outcomes."""
    mv = metrics["var"]["0.01"]
    order = ["hist_var", "gaussian_var", "garch_var", "gjr_var", "fhs_var", "split_var",
             "rolling_var", "dtaci_var", "aci_var"]  # fmt: skip
    order = [m for m in order if m in mv]
    names = {"hist_var": "Historical simulation", "gaussian_var": "Gaussian EWMA (RiskMetrics)",
             "garch_var": "GARCH(1,1)-t", "gjr_var": "GJR-GARCH-t",
             "fhs_var": "Filtered historical sim. (FHS)", "split_var": "Split conformal",
             "rolling_var": "Rolling conformal", "dtaci_var": "DtACI",
             "aci_var": "ACI"}  # fmt: skip
    conformal_set = ("split_var", "rolling_var", "aci_var", "dtaci_var")
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(order) + 1.4))
    ax.axvline(0.01, color=t["muted"], linestyle=(0, (4, 3)), lw=1)
    ax.text(0.0103, -0.75, "target 1%", color=t["ink2"], fontsize=8.5, ha="left")
    for i, m in enumerate(order):
        v = mv[m]
        conformal = m in conformal_set
        c = t["series"][0] if conformal else t["muted"]
        ax.plot([0.01, v["breach_rate"]], [i, i], color=t["grid"], lw=2)
        ax.scatter(v["breach_rate"], i, s=60, color=c, zorder=3)
        k_ok, c_ok = v["kupiec_p"] > 0.05, v["christoffersen_p"] > 0.05
        ax.text(0.0215, i, f"{v['breach_rate']:.2%}", va="center", fontsize=8.8, color=t["ink"])
        ax.text(0.0255, i, "rate " + ("pass" if k_ok else "FAIL"), va="center", fontsize=8.8,
                color=t["good"] if k_ok else t["bad"])  # fmt: skip
        ax.text(0.0295, i, "clustering " + ("pass" if c_ok else "FAIL"), va="center",
                fontsize=8.8, color=t["good"] if c_ok else t["bad"])  # fmt: skip
    ax.set_yticks(range(len(order)), [names[m] for m in order])
    for lbl, m in zip(ax.get_yticklabels(), order, strict=True):
        lbl.set_color(t["ink"] if m in conformal_set else t["ink2"])
    ax.set_xlim(0.004, 0.038)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.set_xticks([0.005, 0.01, 0.015, 0.02])
    style(ax, t, "99% one-day VaR: breach rate, 2020–2026",
          "Blue: conformal methods. Rate = Kupiec test, clustering = Christoffersen test, "
          "both at 5%.", grid="x")  # fmt: skip
    save(fig, "var", theme)


def ablation_chart(t, theme):
    syn = json.loads((REPORTS / "synthetic" / "ablation.json").read_text())
    base = syn[0]["rmse"]
    labels = ["Noise\n(SNR 0)", "SNR 0.5", "SNR 2", "Near noise-free\n(SNR 100)",
              "Oracle\n(true shocks)"]  # fmt: skip
    vals = [r["rmse"] / base - 1 for r in syn[1:]]
    real = {r["variant"]: r["rmse"] for r in json.loads(
        (REPORTS / "default" / "ablation.json").read_text())}  # fmt: skip
    rbase = real["prices only"]
    rows = [("Lexicon", real["+ text (lexicon)"] / rbase - 1)]
    ol = REPORTS / "ollama" / "ablation.json"
    if ol.exists():
        o = {r["variant"]: r["rmse"] for r in json.loads(ol.read_text())}
        rows.append(("SLM (qwen2.5:3b)", o["+ text (ollama)"] / rbase - 1))
    rows.append(("SLM, prompt v1\n(before fixes)", 0.167))  # recorded in docs/PLAN.md, M5

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [5, 3]})
    for ax, xs, ys, title in ((a1, labels, vals, "Synthetic world: does cleaner text help?"),
                              (a2, [r[0] for r in rows], [r[1] for r in rows],
                               "Real FOMC statements")):  # fmt: skip
        colors = [t["series"][0] if v < 0 else t["neg"] for v in ys]
        ax.bar(range(len(ys)), ys, color=colors, width=0.62)
        ax.axhline(0, color=t["axis"], lw=1)
        for i, v in enumerate(ys):
            ax.text(i, v + (0.004 if v >= 0 else -0.004), f"{v:+.1%}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=t["ink"])  # fmt: skip
        ax.set_xticks(range(len(xs)), xs, fontsize=8.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        style(ax, t, title)
        ax.title.set_fontsize(11.5)
    a1.set_ylabel("vol forecast error vs prices only", color=t["ink2"], fontsize=9)
    a1.set_ylim(-0.035, 0.035)
    a2.set_ylim(-0.035, 0.19)
    fig.text(0.01, 0.99, "Blue: text improved the forecast · red: text made it worse",
             color=t["ink2"], fontsize=9.5, va="top")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "ablation", theme)


def slm_chart(t, theme):
    p1, p2 = REPORTS / "slm_accuracy.json", REPORTS / "slm_accuracy_v2.json"
    if not (p1.exists() and p2.exists()):
        return
    v1, v2 = json.loads(p1.read_text()), json.loads(p2.read_text())
    groups = ["Sign agreement\n(clean headlines)", "Correlation with\ntrue shock, SNR 2",
              "Correlation with\ntrue shock, SNR 100"]  # fmt: skip
    series = {
        "SLM, prompt v1": [v1["100.0"]["slm"]["sign_agree"], v1["2.0"]["slm"]["corr_true_z"],
                           v1["100.0"]["slm"]["corr_true_z"]],
        "SLM, prompt v2": [v2["100.0"]["slm"]["sign_agree"], v2["2.0"]["slm"]["corr_true_z"],
                           v2["100.0"]["slm"]["corr_true_z"]],
        "Lexicon": [v2["100.0"]["lex"]["sign_agree"], v2["2.0"]["lex"]["corr_true_z"],
                    v2["100.0"]["lex"]["corr_true_z"]],
    }  # fmt: skip
    colors = [t["muted"], t["series"][0], t["series"][1]]
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    w = 0.26
    for k, ((name, vals), c) in enumerate(zip(series.items(), colors, strict=True)):
        xs = np.arange(3) + (k - 1) * w
        ax.bar(xs, vals, width=w * 0.9, color=c, label=name)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 0.015, f"{v:.2f}" if v < 0.995 else "1.00", ha="center", fontsize=8,
                    color=t["ink2"])  # fmt: skip
    ax.set_xticks(range(3), groups, fontsize=9)
    ax.set_ylim(0, 1.1)
    leg = ax.legend(
        loc="lower left",
        frameon=False,
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0, 1.0),
        borderaxespad=0.2,
    )
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    style(ax, t, "Direction-named schema fixed the SLM's sign errors")
    ax.title.set_position((0, 1.08))
    save(fig, "slm", theme)


def intervals_2020(res, t, theme):
    vol = res.vol.loc["2020-01-01":"2020-12-31"]
    to_ann = math.sqrt(252) * 100
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    for m, color, alpha, label in (("split_absolute", t["muted"], 0.16, "Split conformal"),
                                   ("aci", t["band"], 0.3, "ACI")):  # fmt: skip
        ax.fill_between(vol.index, np.exp(vol[f"{m}_lo"]) * to_ann, np.exp(vol[f"{m}_hi"]) * to_ann,
                        color=color, alpha=alpha, lw=0, label=f"{label} 95% interval")  # fmt: skip
    realized = np.exp(vol["y"]) * to_ann
    ax.plot(vol.index, realized, color=t["ink"], lw=1.2, label="realized 5-day vol")
    miss = ~np.asarray(is_covered(vol["y"], vol["aci_lo"], vol["aci_hi"]))
    ax.scatter(vol.index[miss], realized[miss], s=14, color=t["bad"], zorder=4,
               label="outside ACI interval")  # fmt: skip
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_yticks([3, 10, 30, 100])
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    style(ax, t, "2020 up close: 95% intervals for 5-day realized volatility",
          "Equal-weight SPY/TLT/GLD/USO, annualized. ACI widens after the March misses, "
          "split conformal cannot.", ylabel=None)  # fmt: skip
    save(fig, "intervals_2020", theme)


def basel_chart(res, t, theme):
    """Rolling 250-day 99% VaR exceptions with Basel zones, for three methods."""
    if res.basel is None:
        return
    b = res.basel.dropna(subset=["aci_exceptions"])
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    zone_colors = {"green": "#0ca30c", "yellow": "#fab219", "red": t["bad"]}
    for lo, hi, z in ((-0.5, 4.5, "green"), (4.5, 9.5, "yellow"), (9.5, 13.5, "red")):
        ax.axhspan(lo, hi, color=zone_colors[z], alpha=0.10 if theme == "light" else 0.16, lw=0)
        ax.text(b.index[-1], (lo + hi) / 2, f"  {z} zone", color=t["ink2"], fontsize=8.5,
                va="center", ha="left")  # fmt: skip
    for m, color, lw, name in (("hist", t["muted"], 1.2, "Historical simulation"),
                               ("gaussian", t["series"][1], 1.4, "Gaussian (RiskMetrics)"),
                               ("aci", t["series"][0], 2.0, "ACI (conformal)")):  # fmt: skip
        ax.plot(b.index, b[f"{m}_exceptions"], color=color, lw=lw, label=name)
    ax.set_ylim(-0.5, 13.5)
    ax.set_xlim(b.index[0], b.index[-1])
    leg = ax.legend(loc="upper center", frameon=False, ncol=3, fontsize=9,
                    bbox_to_anchor=(0.5, -0.08))  # fmt: skip
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    style(ax, t, "Basel traffic light: 99% VaR exceptions over the last 250 days",
          "More than 4 exceptions puts a bank's model in the yellow zone and raises its capital "
          "multiplier.", ylabel="exceptions")  # fmt: skip
    save(fig, "basel", theme)


def strategies_chart(res, t, theme):
    """Growth of $1 and rolling 21-day volatility: buy & hold vs vol targeting."""
    if res.strategies is None:
        return
    d = res.strategies
    target = res.info["strategy"]["target_vol"]
    spec = (("buy_hold", t["muted"], 1.3, "Buy & hold"),
            ("vt_ewma", t["series"][1], 1.4, "Vol target · EWMA"),
            ("vt_conformal", t["series"][0], 2.0,
             "Vol target · conformal upper bound"))  # fmt: skip
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35})  # fmt: skip
    for n, color, lw, name in spec:
        r = d[f"{n}_net"].fillna(0)
        wealth = (1 + r).cumprod()
        a1.plot(d.index, wealth, color=color, lw=lw, label=name)
        a2.plot(d.index, r.rolling(21).std() * np.sqrt(252), color=color, lw=lw * 0.8)
    a2.axhline(target, color=t["ink2"], linestyle=(0, (4, 3)), lw=1)
    a2.text(d.index[-1], target, f" target {target:.0%}", color=t["ink2"], fontsize=8.5,
            va="bottom", ha="right")  # fmt: skip
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    leg = a1.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    style(a1, t, "Putting it to work: sizing a portfolio with calibrated risk",
          "Growth of $1 after 5 bp costs, 2020–2026. Each rule sets tomorrow's exposure at "
          "today's close.")  # fmt: skip
    style(a2, t, ylabel="21-day vol")
    save(fig, "strategies", theme)


def main():
    res = BacktestResult.load(REPORTS / "default")
    metrics = compute_metrics(res)
    for theme, t in THEMES.items():
        coverage_dots(res, metrics, t, theme)
        rolling_story(res, t, theme)
        year_heatmap(metrics, t, theme)
        var_chart(metrics, t, theme)
        ablation_chart(t, theme)
        slm_chart(t, theme)
        intervals_2020(res, t, theme)
        basel_chart(res, t, theme)
        strategies_chart(res, t, theme)
    print(f"wrote {len(list(OUT.glob('*.svg')))} files to {OUT}")


if __name__ == "__main__":
    main()
