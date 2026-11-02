"""Regulatory-style risk measures: Expected Shortfall, its backtest, Basel traffic light,
market-risk capital, and Euler risk decomposition.

Sign convention: like the VaR functions in :mod:`aegis.baselines`, a *level* is a threshold on
returns (negative in the tail). ``VaR = −var_level`` and ``ES = −es_level`` are positive losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------------------------
# Expected Shortfall estimators
# --------------------------------------------------------------------------------------------


def gaussian_es(sigma: pd.Series, alpha: float, mean: float = 0.0) -> pd.Series:
    """ES level for N(μ, σ²):  μ − σ · φ(z_α) / α   (the mean return given a return below VaR)."""
    z = stats.norm.ppf(alpha)
    return (mean - sigma * stats.norm.pdf(z) / alpha).rename("gaussian_es")


def student_t_es_factor(alpha: float, nu: float) -> float:
    """ES of a *unit-variance* Student-t, as a positive multiple of σ.

    For a standard t_ν with quantile t_α:  ES = f_ν(t_α)/α · (ν + t_α²)/(ν − 1);
    rescaling to unit variance multiplies by sqrt((ν − 2)/ν).
    """
    t_a = stats.t.ppf(alpha, nu)
    es_std = stats.t.pdf(t_a, nu) / alpha * (nu + t_a**2) / (nu - 1)
    return float(es_std * np.sqrt((nu - 2) / nu))


def historical_es(returns: pd.Series, alpha: float, window: int = 250) -> pd.Series:
    """ES level from the last ``window`` returns: mean of returns at or below their α-quantile."""

    def _tail_mean(x: np.ndarray) -> float:
        q = np.quantile(x, alpha)
        return float(x[x <= q].mean())

    return returns.rolling(window).apply(_tail_mean, raw=True).rename("hist_es")


# --------------------------------------------------------------------------------------------
# ES backtest (Acerbi & Székely, 2014)
# --------------------------------------------------------------------------------------------

# Acerbi–Székely report that the Z2 critical values are nearly distribution-free at α = 2.5%:
# about −0.70 at 5% significance and −1.8 at 0.01%. These are the zone thresholds used here.
Z2_YELLOW, Z2_RED = -0.70, -1.80


@dataclass(frozen=True)
class Z2Result:
    z2: float
    breaches: int
    zone: str  # "green" | "yellow" | "red"


def acerbi_szekely_z2(
    returns: np.ndarray, var_level: np.ndarray, es_level: np.ndarray, alpha: float
) -> Z2Result:
    """Z2 = Σ_t X_t·I_t / (T·α·ES_t) + 1,  I_t = 1{X_t < var_level_t},  ES_t = −es_level_t > 0.

    E[Z2] = 0 when VaR and ES are correctly specified; Z2 < 0 means realized tail losses are
    larger than the forecast ES (ES underestimated), Z2 > 0 means ES is conservative.
    """
    x = np.asarray(returns, float)
    v, e = np.asarray(var_level, float), np.asarray(es_level, float)
    ok = np.isfinite(x) & np.isfinite(v) & np.isfinite(e) & (e < 0)
    x, v, es = x[ok], v[ok], -e[ok]
    hit = x < v
    z2 = float(np.sum(x * hit / es) / (len(x) * alpha) + 1.0)
    zone = "red" if z2 < Z2_RED else "yellow" if z2 < Z2_YELLOW else "green"
    return Z2Result(z2, int(hit.sum()), zone)


# --------------------------------------------------------------------------------------------
# Basel traffic light and market-risk capital (Basel 2.5 internal-models approach)
# --------------------------------------------------------------------------------------------

# Plus factor added to the base multiplier 3 for k exceptions in 250 days at 99% VaR.
_PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_zone(exceptions: int) -> str:
    return "green" if exceptions <= 4 else "yellow" if exceptions <= 9 else "red"


def basel_multiplier(exceptions: int) -> float:
    if exceptions <= 4:
        return 3.0
    return 3.0 + _PLUS.get(exceptions, 1.0)


def traffic_light(breach_forecast_day: pd.Series, window: int = 250) -> pd.DataFrame:
    """Rolling Basel traffic light from 1-day 99% VaR breaches.

    ``breach_forecast_day`` is indexed by the day the forecast was made (outcome the next day).
    It is shifted to the outcome day first, so the count on day t uses only outcomes known by t.
    Counts start once a full ``window`` of outcomes exists.
    """
    known = breach_forecast_day.astype(float).shift(1)
    count = known.rolling(window).sum()
    out = pd.DataFrame({"exceptions": count})
    out["zone"] = count.map(lambda k: basel_zone(int(k)) if np.isfinite(k) else None)
    out["multiplier"] = count.map(lambda k: basel_multiplier(int(k)) if np.isfinite(k) else np.nan)
    return out


def market_risk_capital(var10: pd.Series, multiplier: pd.Series, avg_window: int = 60) -> pd.Series:
    """Basel 2.5 VaR capital: max(VaR_t, m_t · mean(VaR over the last 60 days)), with VaR the
    positive 10-day 99% VaR. Days without a multiplier yet are NaN."""
    v = -var10
    return np.maximum(v, multiplier * v.rolling(avg_window).mean()).rename("capital")


# --------------------------------------------------------------------------------------------
# Euler risk decomposition
# --------------------------------------------------------------------------------------------


def risk_contributions(cov: np.ndarray, weights: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Percent contribution of each asset to portfolio volatility (Euler allocation).

    RC_i = w_i (Σw)_i / (wᵀΣw); the RC_i sum to 1 on every day. Because VaR and ES under any
    scale-family model (Gaussian, t, or conformal with a normalized score) are proportional to
    σ_p, the same shares decompose those measures too: component VaR_i = RC_i · VaR.
    """
    w = np.asarray(weights, float)
    sw = np.einsum("tij,tj->ti", cov, w)
    var_p = np.einsum("ti,ti->t", w, sw)
    return w * sw / var_p[:, None]
