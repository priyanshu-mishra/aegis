"""Non-conformal baselines that the conformal methods are compared against.

Each VaR function returns, for every day t, the lower bound ℓ_t on the next-day return r_{t+1}
such that P(r_{t+1} < ℓ_t) should be α. VaR_α = −ℓ_t. All use information up to day t only.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def gaussian_var(sigma: pd.Series, alpha: float, mean: float = 0.0) -> pd.Series:
    """ℓ_t = μ + Φ⁻¹(α) σ̂_t (parametric, e.g. RiskMetrics with EWMA σ̂)."""
    return (mean + stats.norm.ppf(alpha) * sigma).rename("gaussian_var")


def historical_var(returns: pd.Series, alpha: float, window: int = 250) -> pd.Series:
    """ℓ_t = empirical α-quantile of the last ``window`` returns r_{t−window+1} … r_t."""
    return returns.rolling(window).quantile(alpha, interpolation="linear").rename("hist_var")


def garch_forecast(
    returns: pd.Series, train_end: pd.Timestamp, horizon: int = 1, dist: str = "t"
) -> pd.DataFrame:
    """GARCH(1,1) forecast of the next ``horizon`` days' summed return, made at each day t.

    Parameters are estimated on data up to ``train_end`` only, then frozen; the variance
    recursion is filtered forward, so the forecast at t conditions on returns through t but never
    on later data. Multi-day mean and variance are sums of the h-step forecasts (returns are
    serially uncorrelated under the model). Returns columns ``mu``, ``sigma`` (decimal units) and
    ``nu`` (Student-t degrees of freedom; NaN for a normal model).
    """
    from arch import arch_model

    scaled = returns * 100  # arch is better conditioned on percent returns
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(last_obs=train_end, disp="off")
        fc = res.forecast(start=scaled.index[0], horizon=horizon, reindex=True)
    # Columns are h.1 … h.9, or zero-padded h.01 … h.10 once horizon ≥ 10.
    nu = float(res.params["nu"]) if dist == "t" else np.nan
    return pd.DataFrame(
        {
            "mu": fc.mean.sum(axis=1, min_count=horizon) / 100,
            "sigma": np.sqrt(fc.variance.sum(axis=1, min_count=horizon)) / 100,
            "nu": nu,
        }
    )


def _std_quantile(alpha: float, nu: float) -> float:
    """α-quantile of a unit-variance Student-t (normal if ν is NaN)."""
    if np.isnan(nu):
        return float(stats.norm.ppf(alpha))
    return float(stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu))


def garch_var(
    returns: pd.Series, alpha: float, train_end: pd.Timestamp, dist: str = "t", horizon: int = 1
) -> pd.Series:
    """GARCH(1,1) VaR level for the ``horizon``-day return (see :func:`garch_forecast`)."""
    fc = garch_forecast(returns, train_end, horizon, dist)
    q = _std_quantile(alpha, fc["nu"].iloc[0])
    return (fc["mu"] + q * fc["sigma"]).rename("garch_var")


def garch_es(
    returns: pd.Series, alpha: float, train_end: pd.Timestamp, dist: str = "t", horizon: int = 1
) -> pd.Series:
    """GARCH(1,1) Expected Shortfall level: μ − σ · ES-factor of the fitted innovation law."""
    from aegis.risk_measures import student_t_es_factor

    fc = garch_forecast(returns, train_end, horizon, dist)
    nu = fc["nu"].iloc[0]
    if np.isnan(nu):
        factor = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    else:
        factor = student_t_es_factor(alpha, nu)
    return (fc["mu"] - factor * fc["sigma"]).rename("garch_es")


def historical_var_horizon(returns: pd.Series, alpha: float, horizon: int, window: int = 250):
    """Historical-simulation VaR level for ``horizon``-day returns: α-quantile of the last
    ``window`` overlapping h-day returns that have fully ended by day t."""
    hret = returns.rolling(horizon).sum()
    return hret.rolling(window).quantile(alpha, interpolation="linear").rename("hist_var")


def gaussian_interval(
    mu: np.ndarray, resid_std: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Naive two-sided interval μ̂ ± z_{1−α/2} · s, with s the training residual std."""
    z = stats.norm.ppf(1 - alpha / 2)
    return mu - z * resid_std, mu + z * resid_std
