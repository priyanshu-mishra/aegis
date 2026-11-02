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


def garch_var(
    returns: pd.Series, alpha: float, train_end: pd.Timestamp, dist: str = "t"
) -> pd.Series:
    """GARCH(1,1) VaR with parameters estimated on data up to ``train_end`` only.

    After estimation the parameters are frozen and the variance recursion is filtered forward,
    so the forecast for t+1 conditions on returns through t but never on later data.
    """
    from arch import arch_model

    scaled = returns * 100  # arch is better conditioned on percent returns
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(last_obs=train_end, disp="off")
        fc = res.forecast(start=scaled.index[0], horizon=1, reindex=True)
    mu = fc.mean["h.1"]
    sigma = np.sqrt(fc.variance["h.1"])
    q = (
        stats.t.ppf(alpha, res.params["nu"]) * np.sqrt((res.params["nu"] - 2) / res.params["nu"])
        if dist == "t"
        else stats.norm.ppf(alpha)
    )
    return ((mu + q * sigma) / 100).rename("garch_var")


def gaussian_interval(
    mu: np.ndarray, resid_std: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Naive two-sided interval μ̂ ± z_{1−α/2} · s, with s the training residual std."""
    z = stats.norm.ppf(1 - alpha / 2)
    return mu - z * resid_std, mu + z * resid_std
