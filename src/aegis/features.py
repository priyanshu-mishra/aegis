"""Feature matrix X_t and targets y_t for the factor engine.

Row t of :func:`build_features` uses only information available at the close of day t.
Row t of :func:`build_targets` describes what happens *after* t (days t+1 … t+h).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from aegis.data.news import NewsStore
from aegis.data.prices import MarketData
from aegis.risk_engine import ewma_covariance, portfolio_vol

_FLOOR = 1e-4  # floor for log(vol) so a zero-return day doesn't produce −inf


def _log_vol(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(np.maximum(x, _FLOOR))


def har_components(r: pd.Series) -> pd.DataFrame:
    """HAR-RV inputs (Corsi 2009) on a daily-vol scale: last day, last week, last month."""
    sq = r**2
    return pd.DataFrame(
        {
            "log_rv_d": _log_vol(r.abs()),
            "log_rv_w": _log_vol(np.sqrt(sq.rolling(5).mean())),
            "log_rv_m": _log_vol(np.sqrt(sq.rolling(22).mean())),
        }
    )


def build_features(
    market: MarketData,
    ewma_lambda: float = 0.94,
    signals: pd.DataFrame | None = None,
    engine_cov: np.ndarray | None = None,
) -> pd.DataFrame:
    """X_t: HAR components, EWMA vols, vol-index level/change, and optional text signals.

    ``signals`` must already be aggregated per trading day (as-of safe); columns get a
    ``sig_`` prefix. ``engine_cov`` is the factor engine's (shock-adjusted) covariance path; when
    given, its portfolio vol is added as ``log_engine_vol``. Burn-in rows contain NaN and are
    left for the caller to drop.
    """
    idx = market.returns.index
    cov = ewma_covariance(market.returns, ewma_lambda)
    X = har_components(market.portfolio)
    # Weights for day t+1 are decided at the close of t, so they are known when forecasting.
    next_w = market.weights.shift(-1).ffill()
    X["log_ewma_vol"] = _log_vol(pd.Series(portfolio_vol(cov, next_w), index=idx))
    if engine_cov is not None:
        X["log_engine_vol"] = _log_vol(pd.Series(portfolio_vol(engine_cov, next_w), index=idx))
    asset_vol = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    for i, t in enumerate(market.returns.columns):
        X[f"log_ewma_vol_{t}"] = _log_vol(pd.Series(asset_vol[:, i], index=idx))
    for name, col in market.features.items():
        key = name.strip("^").lower()
        X[f"log_{key}"] = np.log(col)
        X[f"dlog_{key}_5"] = np.log(col).diff(5)
    if signals is not None:
        X = X.join(signals.reindex(idx).add_prefix("sig_"))
    return X


def build_targets(market: MarketData, horizon: int, long_horizon: int = 10) -> pd.DataFrame:
    """Targets (looking forward from t):

    * ``log_vol_fwd``: log realised portfolio vol over t+1…t+h, daily scale
      = log sqrt( (1/h) Σ_{i=1}^{h} r²_{p,t+i} )            (target A)
    * ``ret_fwd``: next-day portfolio log return r_{p,t+1}   (target B)
    * ``ret_fwd_long``: summed log return over t+1 … t+long_horizon (regulatory VaR horizon)
    """
    r = market.portfolio
    mean_sq = (r**2).rolling(horizon).mean().shift(-horizon)
    return pd.DataFrame(
        {
            "log_vol_fwd": _log_vol(np.sqrt(mean_sq)),
            "ret_fwd": r.shift(-1),
            "ret_fwd_long": r.rolling(long_horizon).sum().shift(-long_horizon),
        },
        index=r.index,
    )


SIGNAL_FIELDS = (
    "sentiment",
    "vol_premium",
    *(f"shock_{f}" for f in ("rates", "inflation", "growth", "energy", "geopolitics", "credit")),
)


def aggregate_signals(
    parsed: pd.DataFrame,
    store: NewsStore,
    trading_days: pd.DatetimeIndex,
    half_life_days: float = 3.0,
    prior_weight: float = 0.5,
) -> pd.DataFrame:
    """Daily text features from per-item parser output, as-of safe.

    Each item is assigned to the first trading day whose 16:00 New York close is strictly after
    its publication (:meth:`NewsStore.daily_buckets`), so a day's value depends only on items
    assigned to that day or earlier. With decay d = 2^(−1 / half_life):

    * **shock level** per factor, a decayed confidence-weighted mean over items that mention the
      factor (a parser's 0 for an unmentioned factor is not evidence that the shock is 0):

          L_t = N_t / (D_t + prior),  N_t = d·N_{t−1} + Σ w·x,  D_t = d·D_{t−1} + Σ w

      The prior shrinks stale or thinly-sourced levels toward 0. ``abs_shock_*`` = |L_t|.
    * **news flow** (sentiment, vol premium, item count): decayed confidence-weighted sums,
      A_t = d·A_{t−1} + daily_t.
    """
    days = pd.DatetimeIndex(trading_days)
    buckets = store.daily_buckets(days)
    day_of = {item.id: d for d, items in buckets.items() for item in items}
    df = parsed[parsed["id"].isin(day_of)].copy()
    day = df["id"].map(day_of)
    conf = df["confidence"]
    decay = 0.5 ** (1.0 / half_life_days)

    def decayed(values: pd.Series) -> np.ndarray:
        daily = values.groupby(day).sum().reindex(days, fill_value=0.0).to_numpy()
        return lfilter([1.0], [1.0, -decay], daily)

    out: dict[str, np.ndarray] = {}
    for c in ("sentiment", "vol_premium"):
        if c in df:
            out[c] = decayed(conf * df[c])
    out["intensity"] = decayed(pd.Series(1.0, index=df.index))
    for c in [c for c in SIGNAL_FIELDS if c.startswith("shock_") and c in df]:
        w = conf * (df[c] != 0)
        level = decayed(w * df[c]) / (decayed(w) + prior_weight)
        out[c] = level
        out[f"abs_{c}"] = np.abs(level)
    return pd.DataFrame(out, index=days)


def clip_to_training_range(signals: pd.DataFrame, train_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Clip each signal column to its [min, max] over the training dates.

    Text from a new regime (e.g. pandemic or war language after years of routine statements)
    can produce values never seen in training; a linear model would extrapolate them without
    bound. Bounds come from training dates only, so this uses no future information.
    """
    ref = signals.loc[signals.index.intersection(train_index)]
    return signals.clip(lower=ref.min(), upper=ref.max(), axis=1)
