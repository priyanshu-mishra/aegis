"""Does the text layer help? Re-run the backtest with and without text signals.

Variants:

* **prices only**: no text features; the overlay is a calibrated EWMA
* **+ text**: signals parsed by the configured backend
* **+ oracle shocks** (synthetic only): the true latent shocks, an upper bound on what any parser
  could extract
* **SNR sweep** (synthetic only): headlines regenerated at several signal-to-noise ratios
  (100 ≈ noise-free); the prices are identical across SNRs, only the news changes

All variants share the same splits, models and conformal settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from aegis.backtest import BacktestResult, run_backtest
from aegis.config import AegisConfig
from aegis.data import Dataset, load_dataset
from aegis.data.prices import build_market
from aegis.data.synthetic import FACTORS, SyntheticWorld, generate_world
from aegis.report import var_metrics, vol_metrics

SNR_SWEEP = (0.0, 0.5, 2.0, 100.0)  # 100 ≈ noise-free headlines: the parser's ceiling


def oracle_signals(world: SyntheticWorld, index: pd.DatetimeIndex) -> pd.DataFrame:
    """True latent shocks in the same column layout as parsed signals."""
    z = world.latent.reindex(index)
    out = z.add_prefix("shock_")
    for f in FACTORS:
        out[f"abs_shock_{f}"] = z[f].abs()
    return out


def shock_recovery(parsed: pd.DataFrame | None, world: SyntheticWorld) -> float | None:
    """Correlation between the parsed shock on each headline's factor and the true z."""
    if parsed is None or parsed.empty:
        return None
    truth = world.news_truth.set_index("id")
    merged = parsed.set_index("id").join(truth, how="inner")
    vals = [row[f"shock_{row['factor']}"] for _, row in merged.iterrows()]
    if np.std(vals) == 0:
        return 0.0
    return float(np.corrcoef(vals, merged["z"])[0, 1])


def summarize(res: BacktestResult) -> dict:
    vol, info = res.vol, res.info
    a = info["alpha"]
    vm = vol_metrics(vol, ["cqr", "aci", "split_normalized"], a)
    row = {
        "rmse": float(np.sqrt(np.mean((vol["y"] - vol["mu"]) ** 2))),
        **{f"{m}_coverage": vm[m]["coverage"] for m in vm},
        **{f"{m}_width": vm[m]["mean_width"] for m in vm},
        **{f"{m}_winkler": vm[m]["winkler"] for m in vm},
    }
    if 0.01 in res.var:
        v = var_metrics(res.var[0.01], 0.01)
        row["aci_var99_breach"] = v["aci_var"]["breach_rate"]
        row["aci_var99_mean"] = v["aci_var"]["mean_var"]
    lat = info["latency"].get("parse_one_item_uncached")
    row["parse_p50_ms"] = lat["p50_ms"] if lat else None
    return row


def _with(cfg: AegisConfig, **parser) -> AegisConfig:
    return cfg.model_copy(update={"parser": cfg.parser.model_copy(update=parser)})


def run_ablation(cfg: AegisConfig) -> pd.DataFrame:
    rows: list[dict] = []
    ds = load_dataset(cfg)
    base = summarize(run_backtest(_with(cfg, use_signals=False), ds))
    rows.append({"variant": "prices only", **base})

    if ds.truth is None:
        res = run_backtest(_with(cfg, use_signals=True), ds)
        rows.append({"variant": f"+ text ({cfg.parser.backend})", **summarize(res)})
        return pd.DataFrame(rows).set_index("variant")

    d = cfg.data
    for snr in SNR_SWEEP:
        world = generate_world(
            d.tickers, d.feature_tickers[0] if d.feature_tickers else "VOL", d.start,
            d.synthetic.n_days, snr, d.synthetic.news_per_day, cfg.seed,
        )  # fmt: skip
        market = build_market(world.prices, world.features, cfg.portfolio.weighting,
                              cfg.portfolio.weights, "synthetic")  # fmt: skip
        sds = Dataset(market, world.news, world)
        res = run_backtest(_with(cfg, use_signals=True), sds)
        name = f"+ text ({cfg.parser.backend}), SNR {snr:g}"
        recovery = shock_recovery(res.parsed, world)
        rows.append({"variant": name, **summarize(res), "shock_recovery": recovery})
    oracle = run_backtest(cfg, ds, signals=oracle_signals(ds.truth, ds.market.returns.index))
    rows.append({"variant": "+ oracle shocks (upper bound)", **summarize(oracle),
                 "shock_recovery": 1.0})  # fmt: skip
    return pd.DataFrame(rows).set_index("variant")


def _fmt(x, kind):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return {"pct": f"{x:.1%}", "pct2": f"{x:.2%}", "num": f"{x:.3f}", "ms": f"{x:.2f}"}[kind]


def render_ablation(df: pd.DataFrame, cfg: AegisConfig) -> str:
    base = df.iloc[0]
    rows = []
    for name, r in df.iterrows():
        rows.append([
            name, _fmt(r["rmse"], "num"), f"{(r['rmse'] / base['rmse'] - 1):+.1%}",
            _fmt(r["cqr_coverage"], "pct"), _fmt(r["cqr_width"], "num"),
            _fmt(r["cqr_winkler"], "num"), _fmt(r["aci_coverage"], "pct"),
            _fmt(r["aci_width"], "num"), _fmt(r.get("aci_var99_breach"), "pct2"),
            _fmt(r.get("aci_var99_mean"), "pct2"), _fmt(r.get("shock_recovery"), "num"),
        ])  # fmt: skip
    table = tabulate(rows, headers=["Variant", "Vol RMSE", "vs prices", "CQR cov", "CQR width",
                                    "CQR Winkler", "ACI cov", "ACI width", "99% VaR breach",
                                    "Avg 99% VaR", "Shock recovery"],
                     tablefmt="pipe", disable_numparse=True)  # fmt: skip
    synthetic = cfg.data.source == "synthetic"
    notes = [
        "- *Vol RMSE*: out-of-sample error of the point model on log 5-day realized vol.",
        "- *Width*: mean interval width in log-vol units at the same 95% target; narrower at "
        "equal coverage is the benefit text should buy.",
    ]
    if synthetic:
        notes += [
            "- *Shock recovery*: correlation between the parsed shock on each headline's factor "
            "and the true latent shock. It is capped by the SNR of the headlines themselves.",
            "- The lexicon's phrase rules were written with knowledge of the synthetic headline "
            "templates, so on synthetic data it reads the *observed* value almost perfectly; "
            "the remaining gap to the oracle is the headline noise set by SNR.",
        ]
    return "\n".join([f"# Text-layer ablation: {cfg.name}", "", table, "", *notes]) + "\n"


def write_ablation(df: pd.DataFrame, cfg: AegisConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation.json").write_text(
        json.dumps(df.reset_index().to_dict(orient="records"), indent=2, default=str)
    )
    path = out_dir / "ablation.md"
    path.write_text(render_ablation(df, cfg))
    return path
