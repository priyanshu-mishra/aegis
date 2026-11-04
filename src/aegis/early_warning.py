"""A learned early-warning signal for stress, used to choose the regime for conformal VaR.

The VaR layer adapts after a miss, so the first days of a shock get through. This module asks
whether a model can see the shock coming from information available at the close of day t:

* :func:`stress_features`: as-of-safe inputs (short- against long-run volatility, the vol index
  and its premium over realised volatility, cross-asset correlation, drawdown, recent jumps).
* :func:`stress_label`: 1 if any of the next five daily returns would have breached a Gaussian
  99% VaR at the forecast scale (the event the VaR layer is trying not to miss).
* :class:`StressModel`: a standardised, L2-penalised logistic regression fitted on the training
  period only.

The probability is only used to *sort days into regimes* for :class:`RegimeACI`, so a poor model
costs efficiency but cannot break the coverage guarantee inside each regime.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis.data.prices import MarketData

STRESS_Z = -2.326  # standardised return at the Gaussian 1% quantile
LABEL_DAYS = 5


def _rms(r: pd.Series, n: int) -> pd.Series:
    return np.sqrt((r**2).rolling(n).mean())


def stress_features(market: MarketData, sigma: pd.Series) -> pd.DataFrame:
    """Row t uses only returns and index levels up to the close of day t.

    ``sigma`` is the day-t volatility forecast of the VaR layer; yesterday's value standardises
    today's return.
    """
    r, R = market.portfolio, market.returns
    z = r / sigma.shift(1)
    f = pd.DataFrame(index=r.index)
    f["vol_ratio"] = np.log(_rms(r, 5) / _rms(r, 63))
    f["drawdown"] = r.cumsum() - r.cumsum().rolling(63).max()
    f["jump"] = z.abs().rolling(5).max()
    f["down5"] = z.rolling(5).sum()
    f["corr21"] = pd.concat(
        [R[a].rolling(21).corr(R[b]) for a, b in itertools.combinations(R.columns, 2)], axis=1
    ).mean(axis=1)
    if len(market.features.columns):
        v = market.features.iloc[:, 0]
        f["log_vix"] = np.log(v)
        f["dlog_vix_5"] = np.log(v).diff(5)
        # Implied daily volatility against the model's own forecast: the market's fear premium.
        f["vix_premium"] = np.log(v / 100 / np.sqrt(252)) - np.log(sigma)
    return f


def stress_label(ret_next: pd.Series, sigma: pd.Series) -> pd.Series:
    """1 if any of the returns on days t+1 … t+5 is at or below STRESS_Z standard deviations.

    ``ret_next`` is the next-day return aligned to day t. The last four rows are NaN.
    """
    z_next = ret_next / sigma
    worst = z_next[::-1].rolling(LABEL_DAYS, min_periods=LABEL_DAYS).min()[::-1]
    return (worst <= STRESS_Z).astype(float).where(worst.notna())


@dataclass
class StressModel:
    c: float = 0.1  # inverse L2 strength; fixed, not tuned on any test data

    def fit(self, X: pd.DataFrame, label: pd.Series) -> StressModel:
        ok = X.notna().all(axis=1) & label.notna()
        self.columns_ = list(X.columns)
        self.base_rate_ = float(label[ok].mean())
        self.model_ = make_pipeline(
            StandardScaler(), LogisticRegression(C=self.c, max_iter=1000)
        ).fit(X.loc[ok], label[ok])
        return self

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns_])[:, 1]
