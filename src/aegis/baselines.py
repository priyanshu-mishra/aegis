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
    returns: pd.Series, train_end: pd.Timestamp, horizon: int = 1, dist: str = "t", o: int = 0
) -> pd.DataFrame:
    """GARCH(1,1) forecast of the next ``horizon`` days' summed return, made at each day t.

    Parameters are estimated on data up to ``train_end`` only, then frozen; the variance
    recursion is filtered forward, so the forecast at t conditions on returns through t but never
    on later data. Multi-day mean and variance are sums of the h-step forecasts (returns are
    serially uncorrelated under the model). Returns columns ``mu``, ``sigma`` (decimal units) and
    ``nu`` (Student-t degrees of freedom; NaN for a normal model). ``o=1`` gives GJR-GARCH,
    whose extra term lets negative returns raise volatility more than positive ones (leverage).
    """
    from arch import arch_model

    scaled = returns * 100  # arch is better conditioned on percent returns
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)
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
    returns: pd.Series,
    alpha: float,
    train_end: pd.Timestamp,
    dist: str = "t",
    horizon: int = 1,
    o: int = 0,
) -> pd.Series:
    """GARCH(1,1) (or GJR with ``o=1``) VaR level for the ``horizon``-day return."""
    fc = garch_forecast(returns, train_end, horizon, dist, o)
    q = _std_quantile(alpha, fc["nu"].iloc[0])
    return (fc["mu"] + q * fc["sigma"]).rename("garch_var")


def garch_es(
    returns: pd.Series,
    alpha: float,
    train_end: pd.Timestamp,
    dist: str = "t",
    horizon: int = 1,
    o: int = 0,
) -> pd.Series:
    """GARCH(1,1) Expected Shortfall level: μ − σ · ES-factor of the fitted innovation law."""
    from aegis.risk_measures import student_t_es_factor

    fc = garch_forecast(returns, train_end, horizon, dist, o)
    nu = fc["nu"].iloc[0]
    if np.isnan(nu):
        factor = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    else:
        factor = student_t_es_factor(alpha, nu)
    return (fc["mu"] - factor * fc["sigma"]).rename("garch_es")


def fhs(
    returns: pd.Series,
    alpha: float,
    train_end: pd.Timestamp,
    horizon: int = 1,
    window: int = 1000,
) -> pd.DataFrame:
    """Filtered historical simulation (Barone-Adesi et al., 1999): VaR and ES levels.

    A GARCH(1,1) with frozen training parameters filters the returns into standardized
    residuals z_s = (r_s − μ_s) / σ_s, using only one-step forecasts made the day before. The
    empirical α-quantile (and tail mean) of the last ``window`` residuals through day t then
    scales the forecast for the next ``horizon`` days:

        VaR level = μ_h + σ_h · q_α(z),    ES level = μ_h + σ_h · mean(z | z ≤ q_α(z))

    For ``horizon`` > 1 this applies the 1-day residual quantile to the h-day σ (no path
    simulation), an approximation that ignores how fat tails thin out under aggregation.
    """
    one = garch_forecast(returns, train_end, 1, dist="normal")
    z = (returns - one["mu"].shift(1)) / one["sigma"].shift(1)

    def _quant(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.quantile(x, alpha)) if len(x) >= 250 else np.nan

    def _tail(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 250:
            return np.nan
        q = np.quantile(x, alpha)
        return float(x[x <= q].mean())

    qz = z.rolling(window, min_periods=1).apply(_quant, raw=True)
    tz = z.rolling(window, min_periods=1).apply(_tail, raw=True)
    fc = one if horizon == 1 else garch_forecast(returns, train_end, horizon, dist="normal")
    return pd.DataFrame({"var": fc["mu"] + fc["sigma"] * qz, "es": fc["mu"] + fc["sigma"] * tz})


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
