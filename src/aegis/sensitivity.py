"""Sensitivity analysis: how coverage and width respond to the conformal settings.

α, γ and the rolling window are varied on the saved forecasts of a run (no refitting). The EWMA
decay λ changes the features themselves, so each λ is a full backtest re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tabulate import tabulate

from aegis.backtest import BacktestResult, run_backtest
from aegis.config import AegisConfig
from aegis.conformal_ncm import (
    AbsoluteResidual,
    AdaptiveConformal,
    Forecast,
    RollingConformal,
    SplitConformal,
    run_online,
)
from aegis.metrics import rolling_coverage

ALPHAS = (0.01, 0.05, 0.10, 0.20)
GAMMAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
WINDOWS = (100, 250, 500, 1000)
LAMBDAS = (0.90, 0.94, 0.97, 0.99)


def _summary(covered: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    w = np.maximum(hi - lo, 0)
    return {
        "coverage": float(covered.mean()),
        "width": float(w[np.isfinite(w)].mean()),
        "infinite_days": int((~np.isfinite(w)).sum()),
        "worst_quarter": float(np.nanmin(rolling_coverage(covered, 63))),
    }


def conformal_grid(res: BacktestResult) -> dict:
    """Vary α (split and ACI), γ (ACI) and the window (rolling) on the saved forecasts."""
    if res.vol_cal is None:
        raise ValueError("run predates sensitivity support; re-run `backtest`")
    cfg = res.info["config"]["conformal"]
    h = res.info["horizon"]
    ratio = cfg["aci_floor_ratio"]
    fc, yc = Forecast(mu=res.vol_cal["mu"].to_numpy()), res.vol_cal["y"].to_numpy()
    ft, yt = Forecast(mu=res.vol["mu"].to_numpy()), res.vol["y"].to_numpy()
    ncm = AbsoluteResidual()

    def online(method):
        r = run_online(method.fit(fc, yc), ft, yt, feedback_lag=h)
        return _summary(r.covered, r.lo, r.hi)

    out: dict = {"alpha": {}, "gamma": {}, "window": {}}
    for a in ALPHAS:
        lo, hi = SplitConformal(ncm, a).fit(fc, yc).predict(ft)
        floor = a * ratio if ratio else None
        out["alpha"][str(a)] = {
            "split": _summary((yt >= lo) & (yt <= hi), lo, hi),
            "aci": online(AdaptiveConformal(ncm, a, cfg["aci_gamma"], alpha_floor=floor)),
        }
    a = res.info["alpha"]
    for g in GAMMAS:
        floor = a * ratio if ratio else None
        out["gamma"][str(g)] = online(AdaptiveConformal(ncm, a, g, alpha_floor=floor))
    for w in WINDOWS:
        out["window"][str(w)] = online(RollingConformal(ncm, a, window=w))
    return out


def lambda_grid(cfg: AegisConfig) -> dict:
    """Full re-run per EWMA λ: point-model error, ACI coverage/width, Gaussian 99% VaR breaches."""
    out = {}
    for lam in LAMBDAS:
        risk = cfg.risk.model_copy(update={"ewma_lambda": lam})
        parser = cfg.parser.model_copy(update={"use_signals": False})
        c = cfg.model_copy(update={"risk": risk, "parser": parser})
        r = run_backtest(c)
        v = r.vol
        row = {"rmse": float(np.sqrt(np.mean((v["y"] - v["mu"]) ** 2)))}
        if "aci_lo" in v:
            cov = (v["y"] >= v["aci_lo"]) & (v["y"] <= v["aci_hi"])
            row |= {"aci_coverage": float(cov.mean()),
                    "aci_width": float((v["aci_hi"] - v["aci_lo"]).mean())}  # fmt: skip
        if 0.01 in r.var:
            d = r.var[0.01]
            row["gaussian_var99_breach"] = float((d["r_next"] < d["gaussian_var"]).mean())
        out[str(lam)] = row
    return out


def render(grid: dict, lam: dict | None, alpha: float) -> str:
    lines = ["# Sensitivity analysis", "",
             "Vol-interval target (5-day realized vol), absolute-residual score. Coverage is "
             "out-of-sample on the test period; *worst quarter* is the lowest 63-day coverage.",
             "", "## Nominal level α", ""]  # fmt: skip
    rows = []
    for a, v in grid["alpha"].items():
        rows.append([f"{1 - float(a):.0%}", f"{v['split']['coverage']:.1%}",
                     f"{v['aci']['coverage']:.1%}", f"{v['aci']['width']:.3f}",
                     f"{v['aci']['worst_quarter']:.1%}"])  # fmt: skip
    lines.append(tabulate(rows, headers=["Target", "Split coverage", "ACI coverage", "ACI width",
                                         "ACI worst quarter"], tablefmt="pipe",
                          disable_numparse=True))  # fmt: skip
    lines += ["", f"## ACI step size γ (target {1 - alpha:.0%})", ""]
    rows = [[g, f"{v['coverage']:.1%}", f"{v['width']:.3f}", f"{v['worst_quarter']:.1%}",
             v["infinite_days"]] for g, v in grid["gamma"].items()]  # fmt: skip
    lines.append(tabulate(rows, headers=["γ", "Coverage", "Width", "Worst quarter", "∞ days"],
                          tablefmt="pipe", disable_numparse=True))  # fmt: skip
    lines += ["", "## Rolling-conformal window (days)", ""]
    rows = [[w, f"{v['coverage']:.1%}", f"{v['width']:.3f}", f"{v['worst_quarter']:.1%}"]
            for w, v in grid["window"].items()]  # fmt: skip
    lines.append(tabulate(rows, headers=["Window", "Coverage", "Width", "Worst quarter"],
                          tablefmt="pipe", disable_numparse=True))  # fmt: skip
    if lam:
        lines += ["", "## EWMA decay λ (full re-run, price-only)", ""]
        rows = [[k, f"{v['rmse']:.3f}", f"{v.get('aci_coverage', np.nan):.1%}",
                 f"{v.get('aci_width', np.nan):.3f}",
                 f"{v.get('gaussian_var99_breach', np.nan):.2%}"]
                for k, v in lam.items()]  # fmt: skip
        lines.append(tabulate(rows, headers=["λ", "Vol RMSE", "ACI coverage", "ACI width",
                                             "Gaussian 99% VaR breaches"], tablefmt="pipe",
                              disable_numparse=True))  # fmt: skip
    return "\n".join(lines) + "\n"


def run_sensitivity(cfg: AegisConfig, out_dir: Path, with_lambda: bool = True) -> Path:
    res = BacktestResult.load(out_dir)
    grid = conformal_grid(res)
    lam = lambda_grid(cfg) if with_lambda else None
    (out_dir / "sensitivity.json").write_text(json.dumps({"grid": grid, "lambda": lam}, indent=2))
    path = out_dir / "sensitivity.md"
    path.write_text(render(grid, lam, res.info["alpha"]))
    return path
