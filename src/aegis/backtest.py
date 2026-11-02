"""Walk-forward backtest: train → calibrate → score the test window one day at a time.

Two targets are evaluated:

* **A, volatility intervals**: log realised portfolio vol over the next h days, two-sided
  intervals from every configured conformal method plus two uncalibrated baselines.
* **B, VaR**: lower bound on next-day portfolio return at each level in
  ``conformal.var_alphas``, from conformal methods (split / rolling / ACI on the lower-tail score)
  and from the Gaussian, historical-simulation and GARCH baselines.

Outputs are per-day series (saved as parquet) plus run metadata; :mod:`aegis.report` turns them
into metrics, figures and a markdown report.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aegis import __version__
from aegis.baselines import (
    garch_es,
    garch_var,
    gaussian_interval,
    gaussian_var,
    historical_var,
    historical_var_horizon,
)
from aegis.config import AegisConfig
from aegis.conformal_ncm import (
    AdaptiveConformal,
    LowerTailResidual,
    RollingConformal,
    SplitConformal,
    build_method,
    run_online,
)
from aegis.data import Dataset, load_dataset
from aegis.features import (
    aggregate_signals,
    build_features,
    build_targets,
    clip_to_training_range,
)
from aegis.latency import StageTimer, time_calls
from aegis.risk_engine import (
    VolForecaster,
    apply_shock_overlay,
    ewma_covariance,
    fit_shock_overlay,
    return_forecast,
)
from aegis.risk_measures import (
    gaussian_es,
    historical_es,
    market_risk_capital,
    risk_contributions,
    traffic_light,
)
from aegis.slm_parser import build_parser, parse_items

log = logging.getLogger(__name__)

VAR_METHODS = ("split_var", "rolling_var", "aci_var")
OVERLAY_EXTRA = ("vol_premium", "intensity")  # signal columns the overlay uses besides |shock|


@dataclass
class Splits:
    train: pd.DatetimeIndex
    cal: pd.DatetimeIndex
    test: pd.DatetimeIndex


def make_splits(
    index: pd.DatetimeIndex,
    train_end: dt.date,
    cal_end: dt.date,
    test_end: dt.date | None,
    embargo: int,
) -> Splits:
    """Chronological train / calibration / test split with an embargo after each segment.

    The last ``embargo`` rows of train and of calibration are dropped: their h-day targets
    reach into the next segment, so keeping them would leak future outcomes across the boundary.
    """
    idx = pd.DatetimeIndex(index)
    tr_end, c_end = pd.Timestamp(train_end), pd.Timestamp(cal_end)
    train = idx[idx <= tr_end]
    cal = idx[(idx > tr_end) & (idx <= c_end)]
    test = idx[idx > c_end]
    if test_end is not None:
        test = test[test <= pd.Timestamp(test_end)]
    if embargo:
        train, cal = train[:-embargo], cal[:-embargo]
    if min(len(train), len(cal), len(test)) < 50:
        raise ValueError(
            f"splits too small (train={len(train)}, cal={len(cal)}, test={len(test)}); "
            "check splits.* dates against the data range"
        )
    return Splits(train, cal, test)


@dataclass
class BacktestResult:
    vol: pd.DataFrame  # test days: y, ewma_log_vol, and <method>_lo / <method>_hi columns
    var: dict[float, pd.DataFrame]  # per α: r_next and <method> lower-bound columns
    alpha_paths: pd.DataFrame  # ACI α_t per test day
    info: dict = field(default_factory=dict)  # config, data summary, splits, latency
    parsed: pd.DataFrame | None = None  # per-item parser output (not saved)
    es: pd.DataFrame | None = None  # r_next, <m>_var and <m>_es levels at the ES tail
    var_long: pd.DataFrame | None = None  # r_fwd_long and <m> long-horizon VaR levels
    basel: pd.DataFrame | None = None  # <m>_exceptions, <m>_multiplier, <m>_capital
    contrib: pd.DataFrame | None = None  # per-asset share of portfolio risk

    _EXTRA = ("es", "var_long", "basel", "contrib")

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.vol.to_parquet(out_dir / "vol_series.parquet")
        for a, df in self.var.items():
            df.to_parquet(out_dir / f"var_series_{a:g}.parquet")
        self.alpha_paths.to_parquet(out_dir / "aci_alpha.parquet")
        for name in self._EXTRA:
            df = getattr(self, name)
            if df is not None:
                df.to_parquet(out_dir / f"{name}.parquet")
        (out_dir / "run_info.json").write_text(json.dumps(self.info, indent=2, default=str))

    @classmethod
    def load(cls, out_dir: Path) -> BacktestResult:
        info = json.loads((out_dir / "run_info.json").read_text())
        var = {
            a: pd.read_parquet(out_dir / f"var_series_{a:g}.parquet") for a in info["var_alphas"]
        }
        extra = {
            name: pd.read_parquet(out_dir / f"{name}.parquet")
            for name in cls._EXTRA
            if (out_dir / f"{name}.parquet").exists()
        }
        return cls(
            vol=pd.read_parquet(out_dir / "vol_series.parquet"),
            var=var,
            alpha_paths=pd.read_parquet(out_dir / "aci_alpha.parquet"),
            info=info,
            **extra,
        )


def _var_method(name: str, alpha: float, cfg: AegisConfig, overlap: int = 1):
    """VaR calibrators. ``overlap`` = h for h-day targets sampled daily: each event is then
    seen h times by ACI, so its step is scaled to γ/h."""
    ncm = LowerTailResidual()
    c = cfg.conformal
    floor = alpha * c.aci_floor_ratio if c.aci_floor_ratio else None
    return {
        "split_var": lambda: SplitConformal(ncm, alpha),
        "rolling_var": lambda: RollingConformal(ncm, alpha, window=c.rolling_window),
        "aci_var": lambda: AdaptiveConformal(
            ncm, alpha, gamma=c.aci_gamma / overlap, alpha_floor=floor
        ),
    }[name]()


def text_signals(cfg: AegisConfig, ds: Dataset, timer: StageTimer) -> tuple[pd.DataFrame, dict]:
    """Parse the news store and aggregate it into daily, as-of-safe signal columns."""
    parser = build_parser(cfg.parser, list(ds.market.returns.columns))
    try:
        with timer.stage("parse_news"):
            parsed = parse_items(parser, ds.news, mask=cfg.parser.mask_entities)
    finally:
        parser.close()
    for ms in parser.latencies_ms:
        timer.record("parse_one_item_uncached", ms / 1000)
    signals = aggregate_signals(parsed, ds.news, ds.market.returns.index,
                                cfg.parser.decay_half_life_days)  # fmt: skip
    stats = {
        "backend": parser.inner.name,
        "model": parser.inner.model,
        "items": len(parsed),
        "cache_hits": parser.hits,
        "parsed_now": parser.misses,
        "fallbacks": getattr(parser.inner, "fallbacks", 0),
        "retries": getattr(parser.inner, "retries", 0),
    }
    return signals, stats | {"_parsed": parsed}


def run_backtest(
    cfg: AegisConfig, dataset: Dataset | None = None, signals: pd.DataFrame | None = None
) -> BacktestResult:
    """Run the full backtest. ``signals`` overrides text parsing (e.g. oracle signals)."""
    timer = StageTimer()
    with timer.stage("load_data"):
        ds = dataset if dataset is not None else load_dataset(cfg)
    m, h, alpha = ds.market, cfg.risk.horizon_days, cfg.conformal.alpha

    text_stats: dict = {"used": False}
    parsed = None
    if signals is not None:
        text_stats = {"used": True, "backend": "injected"}
    elif cfg.parser.use_signals and len(ds.news):
        signals, stats = text_signals(cfg, ds, timer)
        parsed = stats.pop("_parsed")
        text_stats = {"used": True, **stats}

    with timer.stage("features"):
        # Factor engine: EWMA covariance with the semantic shock overlay, fit on training dates
        # only. Without text the overlay is fit on zero signals, i.e. a calibrated EWMA, so that
        # the with/without-text comparison isolates the text.
        sp0 = make_splits(m.returns.index, cfg.splits.train_end, cfg.splits.cal_end,
                          cfg.splits.test_end, cfg.splits.embargo_days)  # fmt: skip
        if signals is not None and cfg.parser.clip_to_train:
            signals = clip_to_training_range(signals.reindex(m.returns.index), sp0.train)
        cov = ewma_covariance(m.returns, cfg.risk.ewma_lambda)
        overlay_sig = (
            signals[[c for c in signals.columns if c.startswith("abs_") or c in OVERLAY_EXTRA]]
            if signals is not None and cfg.risk.use_semantic_overlay
            else pd.DataFrame({"none": 0.0}, index=m.returns.index)
        )
        train_mask = m.returns.index.isin(sp0.train)
        overlay = fit_shock_overlay(cov, m.returns, overlay_sig, train_mask, h)
        engine_cov = apply_shock_overlay(
            cov, overlay_sig.reindex(m.returns.index).to_numpy(), overlay
        )
        X = build_features(m, cfg.risk.ewma_lambda, signals=signals, engine_cov=engine_cov)
        Y = build_targets(m, h, cfg.conformal.long_horizon)
        usable = X.notna().all(axis=1) & Y[["log_vol_fwd", "ret_fwd"]].notna().all(axis=1)
        X, Y = X[usable], Y[usable]
    sp = make_splits(X.index, cfg.splits.train_end, cfg.splits.cal_end, cfg.splits.test_end,
                     cfg.splits.embargo_days)  # fmt: skip
    y = Y["log_vol_fwd"]

    # ---- Target A: volatility intervals --------------------------------------------------
    with timer.stage("fit_point_model"):
        model = VolForecaster(cfg.risk.point_model, alpha, h, seed=cfg.seed)
        model.fit(X.loc[sp.train], y.loc[sp.train])
    f_cal, f_test = model.predict(X.loc[sp.cal]), model.predict(X.loc[sp.test])
    y_cal, y_test = y.loc[sp.cal].to_numpy(), y.loc[sp.test].to_numpy()

    vol = pd.DataFrame(
        {"y": y_test, "mu": f_test.mu, "ewma_log_vol": X.loc[sp.test, "log_ewma_vol"]},
        index=sp.test,
    )
    vol["gaussian_lo"], vol["gaussian_hi"] = gaussian_interval(f_test.mu, model.oof_rmse_, alpha)
    vol["raw_quantile_lo"], vol["raw_quantile_hi"] = f_test.lo, f_test.hi
    alpha_paths = pd.DataFrame(index=sp.test)

    c = cfg.conformal
    method = None
    for name in c.methods:
        if name == "one_sided_var":
            continue  # handled in target B
        with timer.stage("calibrate"):
            method = build_method(name, alpha, rolling_window=c.rolling_window,
                                  aci_gamma=c.aci_gamma, base_ncm=c.online_base,
                                  aci_floor_ratio=c.aci_floor_ratio)  # fmt: skip
            method.fit(f_cal, y_cal)
        with timer.stage("walk_forward"):
            res = run_online(method, f_test, y_test, feedback_lag=h)
        vol[f"{name}_lo"], vol[f"{name}_hi"] = res.lo, res.hi
        if res.alpha_path is not None:
            alpha_paths[f"{name}_vol"] = res.alpha_path

    # Per-day scoring latency: one model prediction plus one conformal interval, using the last
    # calibrated method (online methods are the slowest: their quantile is recomputed each day).
    if method is not None:
        X_test, last = X.loc[sp.test], method
        time_calls(lambda i: last.predict(model.predict(X_test.iloc[i : i + 1])),
                   min(200, len(X_test)), timer, "score_one_day")  # fmt: skip

    # ---- Target B: VaR -------------------------------------------------------------------
    sigma = np.exp(X["log_ewma_vol"])
    r_next = Y["ret_fwd"]
    var: dict[float, pd.DataFrame] = {}
    if "one_sided_var" in c.methods:
        f_ret_cal = return_forecast(X.loc[sp.cal, "log_engine_vol"])
        f_ret_test = return_forecast(X.loc[sp.test, "log_engine_vol"])
        for a in c.var_alphas:
            df = pd.DataFrame({"r_next": r_next.loc[sp.test]}, index=sp.test)
            for name in VAR_METHODS:
                meth = _var_method(name, a, cfg).fit(f_ret_cal, r_next.loc[sp.cal].to_numpy())
                with timer.stage("walk_forward"):
                    res = run_online(meth, f_ret_test, df["r_next"].to_numpy(), feedback_lag=1)
                df[name] = res.lo
                if res.alpha_path is not None:
                    alpha_paths[f"{name}_{a:g}"] = res.alpha_path
            with timer.stage("baselines"):
                df["gaussian_var"] = gaussian_var(sigma, a).loc[sp.test]
                df["hist_var"] = historical_var(m.portfolio, a).reindex(sp.test)
                df["garch_var"] = garch_var(m.portfolio, a, sp.train[-1]).reindex(sp.test)
            var[a] = df

    es_df = var_long = basel = None
    if "one_sided_var" in c.methods:
        with timer.stage("risk_measures"):
            es_df = _expected_shortfall(cfg, m, X, r_next, sigma, sp, f_ret_cal, f_ret_test)
            var_long = _long_horizon_var(cfg, m, X, Y, sp)
            if c.long_alpha in var:
                basel = _basel(var[c.long_alpha], var_long, c.basel_window)

    # Euler risk decomposition from the engine covariance (known at the close of each day).
    next_w = m.weights.shift(-1).ffill()
    rc = risk_contributions(engine_cov, next_w.to_numpy())
    contrib = pd.DataFrame(rc, index=m.returns.index, columns=m.returns.columns).loc[sp.test]

    info = {
        "aegis_version": __version__,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": cfg.model_dump(mode="json"),
        "data": {
            "source": m.source,
            "tickers": list(m.returns.columns),
            "first_day": str(m.trading_days[0].date()),
            "last_day": str(m.trading_days[-1].date()),
            "news_items": len(ds.news),
        },
        "text": text_stats,
        "overlay": {
            "beta": overlay.beta.tolist(),
            "c": overlay.c.tolist(),
            "signals": list(overlay_sig.columns),
        },
        "splits": {
            k: {"start": str(v[0].date()), "end": str(v[-1].date()), "rows": len(v)}
            for k, v in (("train", sp.train), ("cal", sp.cal), ("test", sp.test))
        },
        "model": {"kind": model.kind, "features": model.columns_, "oof_rmse": model.oof_rmse_},
        "alpha": alpha,
        "horizon": h,
        "var_alphas": list(var),
        "vol_methods": ["gaussian", "raw_quantile"]
        + [n for n in c.methods if n != "one_sided_var"],
        "latency": timer.summary(),
    }
    info["risk_measures"] = {
        "es_alpha": c.es_alpha,
        "long_horizon": c.long_horizon,
        "long_alpha": c.long_alpha,
        "basel_window": c.basel_window,
    }
    return BacktestResult(
        vol=vol,
        var=var,
        alpha_paths=alpha_paths,
        info=info,
        parsed=parsed,
        es=es_df,
        var_long=var_long,
        basel=basel,
        contrib=contrib,
    )


RISK_METHODS = ("split", "rolling", "aci", "gaussian", "hist", "garch")


def _expected_shortfall(cfg, m, X, r_next, sigma, sp, f_ret_cal, f_ret_test) -> pd.DataFrame:
    """VaR and ES levels at the ES tail (default 2.5%) for every method.

    Conformal ES = μ̂ − σ̂ · (mean of calibration scores at or beyond q̂): the average standardized
    loss beyond VaR in the calibration window. Unlike the VaR bound it carries no finite-sample
    guarantee; it is judged by the Acerbi–Székely backtest like every other method.
    """
    a = cfg.conformal.es_alpha
    out = pd.DataFrame({"r_next": r_next.loc[sp.test]}, index=sp.test)
    y_cal = r_next.loc[sp.cal].to_numpy()
    for name in VAR_METHODS:
        meth = _var_method(name, a, cfg).fit(f_ret_cal, y_cal)
        res = run_online(meth, f_ret_test, out["r_next"].to_numpy(), feedback_lag=1,
                         record_tail=True)  # fmt: skip
        key = name.removesuffix("_var")
        out[f"{key}_var"] = res.lo
        out[f"{key}_es"] = f_ret_test.mu - res.tail * f_ret_test.sigma
    out["gaussian_var"] = gaussian_var(sigma, a).loc[sp.test]
    out["gaussian_es"] = gaussian_es(sigma, a).loc[sp.test]
    out["hist_var"] = historical_var(m.portfolio, a).reindex(sp.test)
    out["hist_es"] = historical_es(m.portfolio, a).reindex(sp.test)
    out["garch_var"] = garch_var(m.portfolio, a, sp.train[-1]).reindex(sp.test)
    out["garch_es"] = garch_es(m.portfolio, a, sp.train[-1]).reindex(sp.test)
    return out


def _long_horizon_var(cfg, m, X, Y, sp) -> pd.DataFrame:
    """VaR for the summed return over the next ``long_horizon`` days (Basel: 10-day 99%).

    Calibration rows whose h-day target overlaps the test period are dropped (an extra embargo
    when h exceeds ``splits.embargo_days``), and outcomes are revealed to online methods only
    h days later.
    """
    c = cfg.conformal
    h, a = c.long_horizon, c.long_alpha
    y = Y["ret_fwd_long"]
    pos = pd.Series(np.arange(len(X.index)), index=X.index)
    first_test = pos[sp.test[0]]
    cal = sp.cal[(pos[sp.cal] + h < first_test).to_numpy()]
    test = sp.test[y.loc[sp.test].notna().to_numpy()]
    scale = np.sqrt(h)  # σ̂ for h days; constant factors cancel in the normalized score anyway
    f_cal = return_forecast(X.loc[cal, "log_engine_vol"] + np.log(scale))
    f_test = return_forecast(X.loc[test, "log_engine_vol"] + np.log(scale))
    out = pd.DataFrame({"r_fwd_long": y.loc[test]}, index=test)
    for name in VAR_METHODS:
        meth = _var_method(name, a, cfg, overlap=h).fit(f_cal, y.loc[cal].to_numpy())
        res = run_online(meth, f_test, out["r_fwd_long"].to_numpy(), feedback_lag=h)
        out[name.removesuffix("_var")] = res.lo
    out["gaussian"] = gaussian_var(np.exp(X["log_ewma_vol"]) * scale, a).loc[test]
    out["hist"] = historical_var_horizon(m.portfolio, a, h).reindex(test)
    out["garch"] = garch_var(m.portfolio, a, sp.train[-1], horizon=h).reindex(test)
    return out


def _basel(var99: pd.DataFrame, var_long: pd.DataFrame, window: int) -> pd.DataFrame:
    """Traffic light from 1-day 99% VaR exceptions; capital from the long-horizon VaR."""
    names = {"split": "split_var", "rolling": "rolling_var", "aci": "aci_var",
             "gaussian": "gaussian_var", "hist": "hist_var", "garch": "garch_var"}  # fmt: skip
    out = pd.DataFrame(index=var99.index)
    for key, col in names.items():
        breach = (var99["r_next"] < var99[col]).where(var99[col].notna())
        tl = traffic_light(breach.fillna(False), window)
        out[f"{key}_exceptions"] = tl["exceptions"]
        out[f"{key}_multiplier"] = tl["multiplier"]
        out[f"{key}_capital"] = market_risk_capital(var_long[key].reindex(out.index),
                                                    tl["multiplier"])  # fmt: skip
    return out
