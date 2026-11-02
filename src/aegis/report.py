"""Turn saved backtest series into metrics, figures and a markdown report.

Every number in the report is computed here from the saved series; nothing is hand-written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from aegis.backtest import RISK_METHODS, VAR_METHODS, BacktestResult
from aegis.metrics import (
    christoffersen_conditional,
    christoffersen_independence,
    coverage_by_group,
    interval_score,
    is_covered,
    kupiec_pof,
    rolling_coverage,
    widths,
)
from aegis.risk_measures import acerbi_szekely_z2, basel_zone

ROLL = 63  # ~one quarter of trading days

LABELS = {
    "gaussian": "Gaussian (uncalibrated)",
    "raw_quantile": "Raw quantile model (uncalibrated)",
    "split_absolute": "Split conformal, absolute residual",
    "split_normalized": "Split conformal, normalized",
    "cqr": "CQR",
    "rolling": "Rolling conformal",
    "aci": "Adaptive conformal (ACI)",
    "dtaci": "Dynamically-tuned ACI (DtACI)",
    "mondrian": "Mondrian conformal (by vol regime)",
    "split_var": "Split conformal VaR",
    "rolling_var": "Rolling conformal VaR",
    "aci_var": "ACI VaR",
    "dtaci_var": "DtACI VaR",
    "gjr_var": "GJR-GARCH-t",
    "fhs_var": "Filtered historical simulation",
    "gaussian_var": "Gaussian EWMA (RiskMetrics)",
    "hist_var": "Historical simulation (250d)",
    "garch_var": "GARCH(1,1)-t",
}

# Reference palette (light surface), categorical slots in fixed order + chrome + status.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRITICAL = "#d03b3b"

# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def _finite(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def vol_metrics(vol: pd.DataFrame, methods: list[str], alpha: float) -> dict:
    y = vol["y"].to_numpy()
    tercile = pd.qcut(vol["ewma_log_vol"], 3, labels=["low", "mid", "high"]).astype(str)
    years = vol.index.year.to_numpy()
    out = {}
    for m in methods:
        lo, hi = vol[f"{m}_lo"].to_numpy(), vol[f"{m}_hi"].to_numpy()
        cov = is_covered(y, lo, hi)
        w = widths(lo, hi)
        finite = np.isfinite(w)
        roll = rolling_coverage(cov, ROLL)
        out[m] = {
            "coverage": float(cov.mean()),
            "mean_width": _finite(float(w[finite].mean())) if finite.any() else None,
            "median_width": float(np.median(w)),
            "infinite_days": int((~finite).sum()),
            "winkler": _finite(interval_score(y, lo, hi, alpha)),
            "worst_rolling_coverage": float(np.nanmin(roll)),
            "by_tercile": coverage_by_group(cov, tercile.to_numpy()),
            "by_year": {int(k): v for k, v in coverage_by_group(cov, years).items()},
        }
    return out


def var_metrics(df: pd.DataFrame, alpha: float) -> dict:
    """Breach statistics per VaR method. An infinite VaR (level −∞) is a valid, if useless,
    forecast that is never breached: it counts toward the breach rate and is reported in
    ``infinite_days`` rather than being dropped."""
    out = {}
    r = df["r_next"].to_numpy()
    for m in [c for c in df.columns if c != "r_next"]:
        lvl = df[m].to_numpy()
        ok = ~np.isnan(lvl)
        b = r[ok] < lvl[ok]
        finite = np.isfinite(lvl[ok])
        out[m] = {
            "days": int(ok.sum()),
            "infinite_days": int((~finite).sum()),
            "breach_rate": float(b.mean()),
            "kupiec_p": kupiec_pof(b, alpha).p_value,
            "christoffersen_p": christoffersen_independence(b).p_value,
            "conditional_p": christoffersen_conditional(b, alpha).p_value,
            "mean_var": float(-lvl[ok][finite].mean()),
        }
    return out


RISK_LABELS = {
    "split": "Split conformal", "rolling": "Rolling conformal", "aci": "ACI",
    "gaussian": "Gaussian EWMA (RiskMetrics)", "hist": "Historical simulation (250d)",
    "garch": "GARCH(1,1)-t", "dtaci": "DtACI", "gjr": "GJR-GARCH-t",
    "fhs": "Filtered historical simulation",
}  # fmt: skip


def es_metrics(df: pd.DataFrame, alpha: float) -> dict:
    """Breach rate at the ES tail, mean ES, and the Acerbi–Székely Z2 statistic per method."""
    out = {}
    r = df["r_next"].to_numpy()
    for m in RISK_METHODS:
        if f"{m}_es" not in df:
            continue
        v, e = df[f"{m}_var"].to_numpy(), df[f"{m}_es"].to_numpy()
        ok = np.isfinite(v) & np.isfinite(e)
        z = acerbi_szekely_z2(r[ok], v[ok], e[ok], alpha)
        out[m] = {"breach_rate": float(np.mean(r[ok] < v[ok])), "mean_es": float(-e[ok].mean()),
                  "z2": z.z2, "zone": z.zone, "days": int(ok.sum())}  # fmt: skip
    return out


def long_var_metrics(df: pd.DataFrame, alpha: float, horizon: int) -> dict:
    """Long-horizon VaR. Overlapping h-day windows make consecutive breaches dependent, so the
    Kupiec and Christoffersen tests use every h-th day only (non-overlapping windows)."""
    out = {}
    r = df["r_fwd_long"].to_numpy()
    for m in RISK_METHODS:
        if m not in df:
            continue
        lvl = df[m].to_numpy()
        ok = ~np.isnan(lvl)
        b_all = r[ok] < lvl[ok]
        b = b_all[::horizon]
        out[m] = {"breach_rate": float(b_all.mean()), "breach_rate_nonoverlap": float(b.mean()),
                  "n_nonoverlap": len(b), "kupiec_p": kupiec_pof(b, alpha).p_value,
                  "christoffersen_p": christoffersen_independence(b).p_value,
                  "infinite_days": int(np.isinf(lvl[ok]).sum()),
                  "mean_var": float(-lvl[ok][np.isfinite(lvl[ok])].mean())}  # fmt: skip
    return out


def basel_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for m in RISK_METHODS:
        if f"{m}_exceptions" not in df:
            continue
        k = df[f"{m}_exceptions"].dropna()
        zones = k.map(lambda x: basel_zone(int(x)))
        cap = df[f"{m}_capital"].dropna()
        out[m] = {
            "days": len(k),
            "green": float((zones == "green").mean()), "yellow": float((zones == "yellow").mean()),
            "red": float((zones == "red").mean()), "max_exceptions": int(k.max()),
            "final_exceptions": int(k.iloc[-1]), "final_zone": zones.iloc[-1],
            "mean_multiplier": float(df[f"{m}_multiplier"].dropna().mean()),
            "mean_capital": float(cap.mean()) if len(cap) else None,
        }  # fmt: skip
    return out


STRATEGY_LABELS = {
    "buy_hold": "Buy & hold (equal weight)",
    "vt_ewma": "Vol target · EWMA vol", "vt_point": "Vol target · point forecast",
    "vt_conformal": "Vol target · conformal upper bound",
    "var_limit_conformal": "VaR limit · conformal VaR",
    "var_limit_gaussian": "VaR limit · Gaussian VaR",
    "alloc_equal": "Equal weight · vol target", "alloc_invvol": "Inverse vol · vol target",
    "alloc_erc": "Risk parity (ERC) · vol target",
}  # fmt: skip


def strategy_table(res: BacktestResult) -> dict:
    from aegis.portfolio import StrategyResult, strategy_metrics

    d, si = res.strategies, res.info["strategy"]
    out = {}
    for n in si["names"]:
        sr = StrategyResult(d[f"{n}_exp"], d[f"{n}_gross"], d[f"{n}_net"], d[f"{n}_turnover"])
        mm = strategy_metrics(sr, si["target_vol"], si["overshoot_window"], si["overshoot_mult"])
        mm["limit_breach_rate"] = float((d[f"{n}_net"].dropna() < -si["var_limit"]).mean())
        out[n] = mm
    return out


def compute_metrics(res: BacktestResult) -> dict:
    info = res.info
    out = {
        "alpha": info["alpha"],
        "vol": vol_metrics(res.vol, info["vol_methods"], info["alpha"]),
        "var": {str(a): var_metrics(df, a) for a, df in res.var.items()},
        "latency": info["latency"],
    }
    rm = info.get("risk_measures")
    if rm and res.es is not None:
        out["es"] = es_metrics(res.es, rm["es_alpha"])
        out["var_long"] = long_var_metrics(res.var_long, rm["long_alpha"], rm["long_horizon"])
    if res.basel is not None:
        out["basel"] = basel_metrics(res.basel)
    if res.strategies is not None:
        out["strategies"] = strategy_table(res)
    if res.contrib is not None:
        c = res.contrib
        out["contrib"] = {"mean": c.mean().to_dict(), "min": c.min().to_dict(),
                          "max": c.max().to_dict()}  # fmt: skip
    return out


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def _to_ann_vol(log_daily: pd.Series | np.ndarray) -> np.ndarray:
    return np.exp(np.clip(np.asarray(log_daily, float), -20, 5)) * np.sqrt(252) * 100


def make_figures(res: BacktestResult, metrics: dict, out_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURFACE})
    files: dict[str, str] = {}
    vol, alpha = res.vol, res.info["alpha"]
    target = 1 - alpha
    methods = res.info["vol_methods"]
    key = [m for m in ("split_absolute", "cqr", "aci") if m in methods][:3]

    # 1. Interval fan charts, one panel per method (small multiples, shared y).
    panels = [m for m in ("split_absolute", "aci") if m in methods]
    if panels:
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), sharex=True,
                                 sharey=True, squeeze=False)  # fmt: skip
        realized = _to_ann_vol(vol["y"])
        y_lo, y_hi = np.nanmin(realized) / 2, np.nanmax(realized) * 2
        for ax, m in zip(axes[:, 0], panels, strict=True):
            lo, hi = _to_ann_vol(vol[f"{m}_lo"]), _to_ann_vol(vol[f"{m}_hi"])
            inf = ~np.isfinite(vol[f"{m}_hi"] - vol[f"{m}_lo"])
            lo, hi = np.where(inf, np.nan, lo), np.where(inf, np.nan, hi)
            ax.fill_between(vol.index, lo, hi, color=SERIES[0], alpha=0.22, linewidth=0,
                            label=f"{int(target * 100)}% interval")  # fmt: skip
            if inf.any():
                label = f"infinite interval ({inf.sum()} days)"
                ax.fill_between(vol.index, y_lo, y_hi, where=inf, color=SERIES[0], alpha=0.08,
                                linewidth=0, label=label)  # fmt: skip
            ax.set_ylim(y_lo, y_hi)
            ax.plot(vol.index, realized, color=INK_2, linewidth=0.8, label="realized")
            miss = ~is_covered(vol["y"], vol[f"{m}_lo"], vol[f"{m}_hi"])
            ax.scatter(vol.index[miss], realized[miss], s=9, color=CRITICAL, zorder=3,
                       label=f"miss ({miss.mean():.1%})")  # fmt: skip
            ax.set_yscale("log")
            _style(ax, LABELS[m], "ann. vol, %")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        fig.tight_layout()
        files["intervals"] = "figures/vol_intervals.png"
        fig.savefig(out_dir / files["intervals"], dpi=140)
        plt.close(fig)

    # 2. Rolling coverage.
    if key:
        fig, ax = plt.subplots(figsize=(10, 3.6))
        y = vol["y"].to_numpy()
        for i, m in enumerate(key):
            cov = is_covered(y, vol[f"{m}_lo"].to_numpy(), vol[f"{m}_hi"].to_numpy())
            roll = rolling_coverage(cov, ROLL)
            ax.plot(vol.index, roll, color=SERIES[i], linewidth=1.6, label=LABELS[m])
        ax.axhline(target, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"target {target:.0%}", (vol.index[0], target), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip
        _style(ax, f"Rolling {ROLL}-day coverage", "coverage")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        fig.tight_layout()
        files["rolling"] = "figures/rolling_coverage.png"
        fig.savefig(out_dir / files["rolling"], dpi=140)
        plt.close(fig)

    # 3. ACI α_t path.
    if "aci_vol" in res.alpha_paths:
        fig, ax = plt.subplots(figsize=(10, 2.8))
        ax.plot(res.alpha_paths.index, res.alpha_paths["aci_vol"], color=SERIES[0], linewidth=1.6)
        ax.axhline(alpha, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"nominal α = {alpha:g}", (res.alpha_paths.index[0], alpha), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip
        _style(ax, "ACI working level α_t (lower → wider intervals)", "α_t")
        fig.tight_layout()
        files["aci_alpha"] = "figures/aci_alpha.png"
        fig.savefig(out_dir / files["aci_alpha"], dpi=140)
        plt.close(fig)

    # 4. Coverage by volatility tercile (grouped bars).
    bar_methods = [m for m in ("gaussian", "split_absolute", "cqr", "aci") if m in methods]
    if bar_methods:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        groups = ["low", "mid", "high"]
        width = 0.8 / len(bar_methods)
        for i, m in enumerate(bar_methods):
            vals = [metrics["vol"][m]["by_tercile"].get(g, np.nan) for g in groups]
            xs = np.arange(3) + (i - (len(bar_methods) - 1) / 2) * width
            ax.bar(xs, vals, width=width * 0.92, color=SERIES[i], label=LABELS[m])
        ax.axhline(target, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(range(3), [f"{g} vol" for g in groups])
        lo_y = min(0.5, min(metrics["vol"][m]["by_tercile"].get(g, 1) for m in bar_methods
                            for g in groups) - 0.05)  # fmt: skip
        ax.set_ylim(lo_y, 1.0)
        _style(ax, "Coverage by forecast-volatility tercile (conditional coverage)", "coverage")
        ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
        fig.tight_layout()
        files["tercile"] = "figures/coverage_by_tercile.png"
        fig.savefig(out_dir / files["tercile"], dpi=140)
        plt.close(fig)

    # 5. VaR at the most extreme level.
    if res.var:
        a = min(res.var)
        df = res.var[a]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(df.index, df["r_next"] * 100, color=AXIS, linewidth=0.6, label="next-day return")
        for i, m in enumerate([c for c in ("aci_var", "gaussian_var") if c in df]):
            ax.plot(df.index, df[m] * 100, color=SERIES[i], linewidth=1.4, label=LABELS[m])
        if "aci_var" in df:
            br = df["r_next"] < df["aci_var"]
            ax.scatter(df.index[br], df["r_next"][br] * 100, s=12, color=CRITICAL, zorder=3,
                       label=f"ACI VaR breach ({br.mean():.2%})")  # fmt: skip
        _style(ax, f"{(1 - a) * 100:g}% one-day VaR", "return, %")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        fig.tight_layout()
        files["var"] = "figures/var.png"
        fig.savefig(out_dir / files["var"], dpi=140)
        plt.close(fig)
    # 6. Basel traffic light: rolling 250-day exceptions for three methods.
    if res.basel is not None:
        b = res.basel
        fig, ax = plt.subplots(figsize=(10, 3.4))
        x = b.index
        ax.axhspan(-0.5, 4.5, color="#0ca30c", alpha=0.08, lw=0)
        ax.axhspan(4.5, 9.5, color="#fab219", alpha=0.12, lw=0)
        ax.axhspan(9.5, 14, color=CRITICAL, alpha=0.08, lw=0)
        for y, label in ((2, "green"), (7, "yellow"), (11.5, "red")):
            ax.text(x[0], y, f" {label}", color=INK_2, fontsize=8, va="center")
        for i, m in enumerate(("aci", "gaussian", "hist")):
            ax.plot(x, b[f"{m}_exceptions"], color=SERIES[i], lw=1.6, label=RISK_LABELS[m])
        ax.set_ylim(-0.5, 14)
        _style(ax, "Basel traffic light: 99% VaR exceptions in the last 250 days",
               "exceptions")  # fmt: skip
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        fig.tight_layout()
        files["basel"] = "figures/basel.png"
        fig.savefig(out_dir / files["basel"], dpi=140)
        plt.close(fig)

    # 8. Strategies: growth of $1 and rolling volatility for the vol-targeting rules.
    if res.strategies is not None:
        d = res.strategies
        names = [n for n in ("buy_hold", "vt_ewma", "vt_conformal") if f"{n}_net" in d]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2]})  # fmt: skip
        colors = {"buy_hold": MUTED, "vt_ewma": SERIES[1], "vt_conformal": SERIES[0]}
        target = res.info["strategy"]["target_vol"]
        for n in names:
            r = d[f"{n}_net"].fillna(0)
            a1.plot(d.index, (1 + r).cumprod(), color=colors[n], lw=1.6, label=STRATEGY_LABELS[n])
            a2.plot(d.index, r.rolling(21).std() * np.sqrt(252), color=colors[n], lw=1.2)
        a2.axhline(target, color=INK_2, ls="--", lw=1)
        a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        _style(a1, "Growth of $1 after costs", "wealth")
        _style(a2, f"Rolling 21-day volatility (target {target:.0%})", "ann. vol")
        a1.legend(loc="upper left", fontsize=8, frameon=False)
        fig.tight_layout()
        files["strategies"] = "figures/strategies.png"
        fig.savefig(out_dir / files["strategies"], dpi=140)
        plt.close(fig)

    # 7. Risk decomposition: signed share of portfolio risk by asset. Shares sum to 100% but can
    # be negative for a hedging asset, so lines (not a stacked area) show them faithfully.
    if res.contrib is not None:
        c = res.contrib
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.axhline(0, color=AXIS, lw=1)
        for i, col in enumerate(c.columns):
            ax.plot(c.index, c[col], color=SERIES[i % len(SERIES)], lw=1.4, label=col)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        _style(ax, "Share of portfolio risk by asset (Euler; sums to 100%, negative = hedge)",
               "share of risk")  # fmt: skip
        ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=c.shape[1])
        fig.tight_layout()
        files["contrib"] = "figures/risk_contributions.png"
        fig.savefig(out_dir / files["contrib"], dpi=140)
        plt.close(fig)
    return files


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------


def _pct(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}%}"


def _num(x, digits=3):
    return "∞" if x is None else f"{x:.{digits}f}"


def _p(x):
    mark = "✓" if x > 0.05 else "✗"
    return f"{x:.3f} {mark}"


def _text_line(info: dict) -> str:
    t = info.get("text", {"used": False})
    if not t.get("used"):
        return f"Text signals: not used ({info['data']['news_items']} news items available)."
    if t.get("backend") == "injected":
        return "Text signals: injected directly (e.g. oracle shocks)."
    return (
        f"Text signals: {t['items']} items parsed by `{t['backend']}` ({t['model']}); "
        f"{t['cache_hits']} from cache, {t['fallbacks']} fallbacks."
    )


def render_markdown(res: BacktestResult, metrics: dict, figures: dict[str, str]) -> str:
    info, alpha = res.info, metrics["alpha"]
    target = 1 - alpha
    sp, d = info["splits"], info["data"]
    vm = metrics["vol"]
    lines = [
        f"# Aegis backtest: {info['config']['name']}",
        "",
        f"Generated {info['created']} by aegis {info['aegis_version']}. "
        "All numbers below are computed from the saved series in this directory.",
        "",
        "## Setup",
        "",
        f"- **Data:** {d['source']}, {', '.join(d['tickers'])} "
        f"({d['first_day']} → {d['last_day']}), {info['config']['portfolio']['weighting']}-weight"
        f" portfolio. {_text_line(info)}",
        f"- **Splits:** train {sp['train']['start']} → {sp['train']['end']} "
        f"({sp['train']['rows']} days) · calibration {sp['cal']['start']} → {sp['cal']['end']} "
        f"({sp['cal']['rows']}) · test {sp['test']['start']} → {sp['test']['end']} "
        f"({sp['test']['rows']}), with a {info['config']['splits']['embargo_days']}-day embargo.",
        f"- **Point model:** {info['model']['kind']} on {len(info['model']['features'])} features "
        f"(out-of-fold RMSE {info['model']['oof_rmse']:.3f} log units).",
        f"- **Conformal:** α = {alpha:g}; online methods use the "
        f"`{info['config']['conformal']['online_base']}` score, outcomes revealed after "
        f"{info['horizon']} days; ACI γ = {info['config']['conformal']['aci_gamma']:g}, "
        f"rolling window {info['config']['conformal']['rolling_window']}.",
        "",
        f"## Target A: {info['horizon']}-day realized volatility intervals",
        "",
        "Width is in log units of daily vol (a width of 0.5 means the upper bound is "
        "e^0.5 ≈ 1.65× the lower), averaged over days with a finite interval; "
        "∞ days counts days where the method returned an unbounded interval (always covered). "
        "Winkler score: lower is better.",
        "",
    ]
    rows = []
    for m, v in vm.items():
        t = v["by_tercile"]
        rows.append([
            LABELS.get(m, m), _pct(v["coverage"]), _num(v["mean_width"]), v["infinite_days"],
            _num(v["winkler"]), _pct(v["worst_rolling_coverage"]), _pct(t.get("low")),
            _pct(t.get("mid")), _pct(t.get("high")),
        ])  # fmt: skip
    lines.append(tabulate(rows, headers=["Method", "Coverage", "Avg width", "∞ days", "Winkler",
                                         f"Worst {ROLL}d", "Low vol", "Mid vol", "High vol"],
                          tablefmt="pipe", disable_numparse=True))  # fmt: skip
    near = [LABELS.get(m, m) for m, v in vm.items() if abs(v["coverage"] - target) <= 0.015]
    lines += [
        "",
        f"Within ±1.5 pp of the {target:.0%} target: "
        + (", ".join(near) if near else "none")
        + ".",
        "",
        "### Coverage by year",
        "",
    ]
    years = sorted({y for v in vm.values() for y in v["by_year"]})
    rows = [[LABELS.get(m, m)] + [_pct(v["by_year"].get(y)) for y in years] for m, v in vm.items()]
    lines.append(tabulate(rows, headers=["Method", *map(str, years)], tablefmt="pipe",
                          disable_numparse=True))  # fmt: skip
    captions = {
        "intervals": "Intervals vs realized vol",
        "rolling": "Rolling coverage",
        "aci_alpha": "ACI α_t",
        "tercile": "Coverage by vol tercile",
    }
    for k, caption in captions.items():
        if k in figures:
            lines += ["", f"![{caption}]({figures[k]})"]

    lines += ["", "## Target B: one-day portfolio VaR", ""]
    lines.append(
        "✓ = test not rejected at 5% significance. Kupiec checks the breach rate; "
        "Christoffersen checks that breaches don't cluster; conditional combines both."
    )
    for a, mv in metrics["var"].items():
        a = float(a)
        lines += ["", f"### {(1 - a) * 100:g}% VaR (target breach rate {a:.0%})", ""]
        rows = []
        order = [m for m in (*VAR_METHODS, "gaussian_var", "hist_var", "garch_var", "gjr_var",
                             "fhs_var") if m in mv]  # fmt: skip
        for m in order:
            v = mv[m]
            rows.append([LABELS.get(m, m), _pct(v["breach_rate"], 2), _p(v["kupiec_p"]),
                         _p(v["christoffersen_p"]), _p(v["conditional_p"]),
                         _pct(v["mean_var"], 2)])  # fmt: skip
        lines.append(tabulate(rows, headers=["Method", "Breaches", "Kupiec p",
                                             "Christoffersen p", "Conditional p", "Avg VaR"],
                              tablefmt="pipe", disable_numparse=True))  # fmt: skip
    if "var" in figures:
        lines += ["", f"![VaR]({figures['var']})"]

    lines += _risk_measures_markdown(res, metrics, figures)
    lines += ["", "## Latency", ""]
    rows = [[k, v["calls"], f"{v['total_s']:.2f}", f"{v['p50_ms']:.2f}", f"{v['p95_ms']:.2f}"]
            for k, v in metrics["latency"].items()]  # fmt: skip
    lines.append(tabulate(rows, headers=["Stage", "Calls", "Total s", "p50 ms", "p95 ms"],
                          tablefmt="pipe", disable_numparse=True))  # fmt: skip

    lines += [
        "",
        "## Guarantees and limitations",
        "",
        "- Split conformal guarantees **marginal** coverage of at least 1 − α only when "
        "calibration and test data are exchangeable. Market data are not, so the split methods "
        "carry no guarantee here; their test coverage is an empirical result.",
        "- ACI's guarantee is deterministic but long-run: the average miss rate converges to α "
        "for any sequence, at rate ~1/(γT). It says nothing about any particular quarter.",
        "- Coverage within volatility terciles is conditional coverage, which no method "
        "here guarantees.",
        "- Statistical tests at 5% significance will reject a correct model 5% of the time.",
    ]
    return "\n".join(lines) + "\n"


def _risk_measures_markdown(res: BacktestResult, metrics: dict, figures: dict) -> list[str]:
    if "es" not in metrics:
        return []
    rm = res.info["risk_measures"]
    a, h, la = rm["es_alpha"], rm["long_horizon"], rm["long_alpha"]
    lines = ["", f"## Expected Shortfall ({(1 - a) * 100:g}%)", "",
             "Basel III / FRTB measures market risk with ES at 97.5%: the average loss on the "
             f"worst {a:.1%} of days. The Acerbi–Székely Z2 statistic is 0 when ES is right and "
             "negative when realized tail losses exceed it (green ≥ −0.70 > yellow ≥ −1.80 > red). "
             "Conformal ES is the mean calibration score beyond the VaR quantile: an empirical "
             "estimate without the coverage guarantee of the VaR bound.", ""]  # fmt: skip
    rows = [[RISK_LABELS[m], _pct(v["breach_rate"], 2), _pct(v["mean_es"], 2), f"{v['z2']:+.2f}",
             v["zone"]] for m, v in metrics["es"].items()]  # fmt: skip
    lines.append(tabulate(rows, headers=["Method", f"Breaches (target {a:.1%})", "Avg ES",
                                         "Z2", "Zone"], tablefmt="pipe",
                          disable_numparse=True))  # fmt: skip

    lines += ["", f"## {h}-day {(1 - la) * 100:g}% VaR (regulatory horizon)", "",
              f"Overlapping {h}-day windows make consecutive breaches dependent, so the tests use "
              f"every {h}th day only.", ""]  # fmt: skip
    rows = [[RISK_LABELS[m], _pct(v["breach_rate"], 2), _pct(v["breach_rate_nonoverlap"], 2),
             v["n_nonoverlap"], _p(v["kupiec_p"]), _p(v["christoffersen_p"]),
             _pct(v["mean_var"], 2)] for m, v in metrics["var_long"].items()]  # fmt: skip
    lines.append(tabulate(rows, headers=["Method", "Breaches", "Breaches (non-overlap)", "n",
                                         "Kupiec p", "Christoffersen p", "Avg VaR"],
                          tablefmt="pipe", disable_numparse=True))  # fmt: skip

    if "basel" in metrics:
        lines += ["", "## Basel traffic light and capital", "",
                  "Exceptions of the 1-day 99% VaR in the last 250 days set the zone and the "
                  "capital multiplier (3.0 in green, up to 4.0 in red). Capital = max(VaR₁₀, "
                  "multiplier × 60-day average VaR₁₀), as a fraction of portfolio value.",
                  ""]  # fmt: skip
        rows = [[RISK_LABELS[m], _pct(v["green"]), _pct(v["yellow"]), _pct(v["red"]),
                 v["max_exceptions"], f"{v['mean_multiplier']:.2f}",
                 _pct(v["mean_capital"], 1)] for m, v in metrics["basel"].items()]  # fmt: skip
        lines.append(tabulate(rows, headers=["Method", "Days green", "Days yellow", "Days red",
                                             "Max exceptions", "Avg multiplier", "Avg capital"],
                              tablefmt="pipe", disable_numparse=True))  # fmt: skip
        if "basel" in figures:
            lines += ["", f"![Basel traffic light]({figures['basel']})"]

    if "strategies" in metrics:
        si = res.info["strategy"]
        lines += ["", "## Putting it to work: position sizing", "",
                  f"Each rule sets the next day's exposure at today's close. Vol-targeting rules "
                  f"aim at {si['target_vol']:.0%} annualized volatility with a scale fitted on "
                  "the calibration period only (max leverage "
                  f"{si['max_leverage']:g}×); VaR-limit rules size so the forecast 99% VaR equals "
                  f"a {si['var_limit']:.0%} daily loss limit. Costs: {si['cost_bps']:g} bp per "
                  f"unit of turnover. Overshoot = share of {si['overshoot_window']}-day windows "
                  f"with realized vol above {si['overshoot_mult']:g}× target.", ""]  # fmt: skip
        rows = []
        for n, v in metrics["strategies"].items():
            rows.append([STRATEGY_LABELS.get(n, n), _pct(v["ann_return"]), _pct(v["ann_vol"]),
                         f"{v['sharpe']:.2f}", _pct(v["max_drawdown"]),
                         _pct(v["overshoot_share"]), _pct(v["limit_breach_rate"], 2),
                         f"{v['avg_exposure']:.2f}", f"{v['annual_turnover']:.1f}",
                         _pct(v["cost_drag"], 2)])  # fmt: skip
        lines.append(tabulate(rows, headers=["Strategy", "Return", "Vol", "Sharpe", "Max DD",
                                             "Overshoot", f"Days loss > {si['var_limit']:.0%}",
                                             "Avg exposure", "Turnover/yr", "Cost/yr"],
                              tablefmt="pipe", disable_numparse=True))  # fmt: skip
        if "strategies" in figures:
            lines += ["", f"![Strategies]({figures['strategies']})"]

    if "contrib" in metrics:
        c = metrics["contrib"]
        lines += ["", "## Risk decomposition", "",
                  "Euler allocation of portfolio volatility from the engine covariance; the same "
                  "shares decompose VaR and ES for scale-family models.", ""]  # fmt: skip
        rows = [[k, _pct(c["mean"][k]), _pct(c["min"][k]), _pct(c["max"][k])] for k in c["mean"]]
        lines.append(tabulate(rows, headers=["Asset", "Average share of risk", "Lowest", "Peak"],
                              tablefmt="pipe", disable_numparse=True))  # fmt: skip
        if "contrib" in figures:
            lines += ["", f"![Risk contributions]({figures['contrib']})"]
    return lines


def write_report(res: BacktestResult, out_dir: Path, figures: bool = True) -> Path:
    metrics = compute_metrics(res)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    figs = make_figures(res, metrics, out_dir) if figures else {}
    path = out_dir / "backtest_report.md"
    path.write_text(render_markdown(res, metrics, figs))
    return path
