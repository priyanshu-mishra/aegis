"""Export saved runs as compact JSON for the static site (``site/public/data``).

Layout is columnar (``{"dates": [...], "series": {name: [...]}}``) so each file is one small
fetch per chapter. Non-finite numbers become ``null``. Inputs that are missing (e.g. the SLM
ablation, which needs Ollama) leave any existing file untouched, so a nightly CI run never
deletes data it cannot regenerate.

``golden.json`` holds reference values computed here in Python; the site's JavaScript
conformal code must reproduce them (checked in the site's own tests).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from aegis import __version__
from aegis.backtest import BacktestResult
from aegis.conformal_ncm import (
    AbsoluteResidual,
    AdaptiveConformal,
    Forecast,
    SplitConformal,
    run_online,
)
from aegis.report import compute_metrics

SCHEMA_VERSION = 1
DIGITS = 5  # log-vol series used by the in-browser conformal code (golden values use these)
LEVEL_DIGITS = 4  # returns, VaR / ES levels and capital: 1 basis point


def _num(x, digits: int = DIGITS):
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


def _col(values, digits: int = DIGITS) -> list:
    return [_num(v, digits) for v in np.asarray(values, dtype=float)]


def _frame(df: pd.DataFrame, digits: int = DIGITS, columns=None) -> dict:
    cols = list(columns) if columns is not None else list(df.columns)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "series": {c: _col(df[c], digits) for c in cols if c in df},
    }


def _clean(obj, digits: int | None = 6):
    """Recursively make JSON-safe: NaN/inf → null, numpy scalars → Python. ``digits=None``
    keeps full float precision."""
    if isinstance(obj, dict):
        return {str(k): _clean(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, digits) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return x if digits is None else round(x, digits)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write(out_dir: Path, name: str, payload, digits: int | None = 6) -> Path:
    path = out_dir / f"{name}.json"
    text = json.dumps(_clean(payload, digits), separators=(",", ":"), allow_nan=False)
    path.write_text(text)
    return path


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None


def golden_values(res: BacktestResult, alphas=(0.05, 0.10)) -> dict:
    """Reference conformal outputs on the exported data (absolute score, as the site uses).

    For each α: the split-conformal threshold q̂ on the calibration scores, and ACI's coverage
    and first/last interval bounds on the test period with the run's γ, floor and feedback lag.
    Inputs are rounded to the exported precision first, so the site can reproduce these
    numbers exactly from the JSON it loads.
    """
    cfg = res.info["config"]["conformal"]
    h = res.info["horizon"]

    def rounded(s: pd.Series) -> np.ndarray:
        return np.round(s.to_numpy(dtype=float), DIGITS)

    fc, yc = Forecast(mu=rounded(res.vol_cal["mu"])), rounded(res.vol_cal["y"])
    ft, yt = Forecast(mu=rounded(res.vol["mu"])), rounded(res.vol["y"])
    out = {"feedback_lag": h, "gamma": cfg["aci_gamma"], "floor_ratio": cfg["aci_floor_ratio"],
           "levels": {}}  # fmt: skip
    for a in alphas:
        split = SplitConformal(AbsoluteResidual(), a).fit(fc, yc)
        floor = a * cfg["aci_floor_ratio"] if cfg["aci_floor_ratio"] else None
        aci = AdaptiveConformal(AbsoluteResidual(), a, cfg["aci_gamma"], alpha_floor=floor)
        r = run_online(aci.fit(fc, yc), ft, yt, feedback_lag=h)
        out["levels"][str(a)] = {
            "split_q": split.q,
            "split_coverage": float(np.mean(np.abs(yt - ft.mu) <= split.q)),
            "aci_coverage": float(r.covered.mean()),
            "aci_first": [r.lo[0], r.hi[0]],
            "aci_last": [r.lo[-1], r.hi[-1]],
            "aci_alpha_last": float(r.alpha_path[-1]),
        }
    return out


def _var_frame(df: pd.DataFrame) -> dict:
    """VaR levels at 1bp, plus a ``<method>_breach`` flag (1/0, null with no forecast) computed at
    full precision as in ``report.var_metrics``; the rounded levels alone cannot settle ties."""
    frame = _frame(df, LEVEL_DIGITS)
    r = df["r_next"].to_numpy(dtype=float)
    for m in [c for c in df.columns if c != "r_next"]:
        lvl = df[m].to_numpy(dtype=float)
        frame["series"][f"{m}_breach"] = [
            None if np.isnan(v) else int(x < v) for x, v in zip(r, lvl, strict=True)
        ]
    return frame


def export_site(
    run_dir: Path,
    out_dir: Path,
    extra_dirs: dict[str, Path] | None = None,
    prices: pd.DataFrame | None = None,
    news: list[dict] | None = None,
) -> list[Path]:
    """Write the site data files from a saved run (``reports/<name>/``).

    ``extra_dirs`` maps labels to other run folders whose ablation / sensitivity / benchmark
    outputs should be included (e.g. ``{"synthetic": reports/synthetic, "ollama": ...}``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    res = BacktestResult.load(run_dir)
    info = res.info
    written: list[Path] = []

    meta = {
        "schema_version": SCHEMA_VERSION,
        "aegis_version": __version__,
        "git_sha": _git_sha(),
        "exported": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "run_created": info["created"],
        "data": info["data"],
        "splits": info["splits"],
        "alpha": info["alpha"],
        "horizon": info["horizon"],
        "model": {k: info["model"][k] for k in ("kind", "oof_rmse")},
        "vol_methods": info["vol_methods"],
        "risk_measures": info.get("risk_measures"),
        "strategy": info.get("strategy"),
        "text": info.get("text"),
    }
    written.append(_write(out_dir, "meta", meta))
    written.append(_write(out_dir, "metrics", compute_metrics(res)))
    written.append(_write(out_dir, "vol", _frame(res.vol)))
    if res.vol_cal is not None:
        written.append(_write(out_dir, "calibration", _frame(res.vol_cal)))
        written.append(_write(out_dir, "golden", golden_values(res), digits=None))
    written.append(_write(out_dir, "aci_alpha", _frame(res.alpha_paths, 5)))
    var = {str(a): _var_frame(df) for a, df in res.var.items()}
    written.append(_write(out_dir, "var", var))
    for name, digits in (("es", LEVEL_DIGITS), ("var_long", LEVEL_DIGITS), ("contrib", 3)):
        df = getattr(res, name)
        if df is not None:
            written.append(_write(out_dir, name, _frame(df, digits)))
    if res.basel is not None:
        b = res.basel.dropna(how="all")
        frame = _frame(b, LEVEL_DIGITS)
        for c in b.columns:
            if c.endswith(("_exceptions", "_multiplier")):
                frame["series"][c] = _col(b[c], 2)
        written.append(_write(out_dir, "basel", frame))
    if res.strategies is not None:
        cols = [c for c in res.strategies.columns if c.endswith(("_exp", "_net"))]
        written.append(
            _write(out_dir, "strategies", _frame(res.strategies, LEVEL_DIGITS + 1, cols))
        )

    for label, path in {"default": run_dir, **(extra_dirs or {})}.items():
        for src, dst in (("ablation.json", f"ablation_{label}"),
                         ("sensitivity.json", f"sensitivity_{label}")):  # fmt: skip
            f = Path(path) / src
            if f.exists():
                written.append(_write(out_dir, dst, json.loads(f.read_text())))
    reports = run_dir.parent
    for src in ("benchmark.json", "slm_accuracy.json", "slm_accuracy_v2.json"):
        f = reports / src
        if f.exists():
            written.append(_write(out_dir, src.removesuffix(".json"), json.loads(f.read_text())))

    if prices is not None:
        # Normalized to 100 at the start, weekly closes: enough for the "meet the portfolio" view.
        weekly = prices.resample("W-FRI").last().dropna(how="all")
        written.append(_write(out_dir, "prices", _frame(100 * weekly / weekly.iloc[0], 2)))
    if news is not None:
        written.append(_write(out_dir, "news", news))
    written.append(_write(out_dir, "lexicon", lexicon_rules()))
    return written


def fomc_digest(store, parser, tickers: list[str]) -> list[dict]:
    """Per FOMC statement: date, title, first paragraph and the parsed shock vector."""
    from aegis.slm_parser import mask_entities

    out = []
    for item in store:
        sig = parser.parse(mask_entities(item.text))
        first = item.body.split("\n\n")[0]
        out.append({
            "id": item.id,
            "published": item.published_at.isoformat(),
            "title": item.title,
            "excerpt": first[:400],
            "shocks": sig.macro_shock_vector.model_dump(by_alias=True),
            "sentiment": sig.sentiment_score,
        })  # fmt: skip
    return out


def lexicon_rules() -> dict:
    """The lexicon parser's rules as plain regex strings (case-insensitive), for a JS port."""
    from aegis import slm_parser as sp

    return {
        "flags": "i",
        "direction": {f: [[p, sgn] for p, sgn in rules] for f, rules in sp._DIRECTION.items()},
        "positive": sp._POSITIVE.pattern,
        "negative": sp._NEGATIVE.pattern,
        "risk_up": sp._RISK_UP.pattern,
        "risk_down": sp._RISK_DOWN.pattern,
        "strong": sp._STRONG.pattern,
        "moderate": sp._MODERATE.pattern,
        "negation": sp._NEGATION.pattern,
        "sentences": sp._SENTENCES.pattern,
        "exposure": sp._EXPOSURE,
        "aliases": {name: fld.alias for name, fld in sp.ShockVector.model_fields.items()},
    }
