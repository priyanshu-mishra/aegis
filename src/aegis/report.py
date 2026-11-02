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

from aegis.backtest import VAR_METHODS, BacktestResult
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

ROLL = 63  # ~one quarter of trading days

LABELS = {
    "gaussian": "Gaussian (uncalibrated)",
    "raw_quantile": "Raw quantile model (uncalibrated)",
    "split_absolute": "Split conformal, absolute residual",
    "split_normalized": "Split conformal, normalized",
    "cqr": "CQR",
    "rolling": "Rolling conformal",
    "aci": "Adaptive conformal (ACI)",
    "split_var": "Split conformal VaR",
    "rolling_var": "Rolling conformal VaR",
    "aci_var": "ACI VaR",
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
    out = {}
    r = df["r_next"].to_numpy()
    for m in [c for c in df.columns if c != "r_next"]:
        lvl = df[m].to_numpy()
        ok = np.isfinite(lvl)
        b = r[ok] < lvl[ok]
        out[m] = {
            "days": int(ok.sum()),
            "breach_rate": float(b.mean()),
            "kupiec_p": kupiec_pof(b, alpha).p_value,
            "christoffersen_p": christoffersen_independence(b).p_value,
            "conditional_p": christoffersen_conditional(b, alpha).p_value,
            "mean_var": float(-lvl[ok].mean()),
        }
    return out


def compute_metrics(res: BacktestResult) -> dict:
    info = res.info
    return {
        "alpha": info["alpha"],
        "vol": vol_metrics(res.vol, info["vol_methods"], info["alpha"]),
        "var": {str(a): var_metrics(df, a) for a, df in res.var.items()},
        "latency": info["latency"],
    }


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
        order = [m for m in (*VAR_METHODS, "gaussian_var", "hist_var", "garch_var") if m in mv]
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


def write_report(res: BacktestResult, out_dir: Path, figures: bool = True) -> Path:
    metrics = compute_metrics(res)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    figs = make_figures(res, metrics, out_dir) if figures else {}
    path = out_dir / "backtest_report.md"
    path.write_text(render_markdown(res, metrics, figs))
    return path
