"""Evaluation metrics for prediction intervals and VaR forecasts."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import xlogy


def is_covered(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Boolean mask ``lo ≤ y ≤ hi`` (an empty interval, lo > hi, never covers)."""
    y = np.asarray(y, dtype=float)
    return (np.asarray(lo) <= y) & (y <= np.asarray(hi))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical coverage: fraction of outcomes inside their interval."""
    return float(np.mean(is_covered(y, lo, hi)))


def widths(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Interval widths, with empty intervals (lo > hi) counted as 0. Infinite widths stay inf."""
    return np.maximum(np.asarray(hi, float) - np.asarray(lo, float), 0.0)


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(widths(lo, hi)))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Mean Winkler interval score (Gneiting & Raftery 2007); lower is better.

    IS = (hi − lo) + (2/α)(lo − y)·1{y < lo} + (2/α)(y − hi)·1{y > hi}

    Rewards narrow intervals but penalises misses in proportion to how far outside y lands.
    """
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    below = np.where(y < lo, lo - y, 0.0)
    above = np.where(y > hi, y - hi, 0.0)
    return float(np.mean(widths(lo, hi) + (2 / alpha) * (below + above)))


def rolling_coverage(covered: np.ndarray, window: int = 63) -> np.ndarray:
    """Trailing-window coverage; the first ``window − 1`` entries are NaN."""
    c = np.asarray(covered, dtype=float)
    out = np.full(c.shape, np.nan)
    if len(c) >= window:
        csum = np.concatenate([[0.0], np.cumsum(c)])
        out[window - 1 :] = (csum[window:] - csum[:-window]) / window
    return out


def coverage_by_group(covered: np.ndarray, groups: Sequence[Hashable]) -> dict[Hashable, float]:
    """Coverage within each group, e.g. volatility terciles or calendar years."""
    covered, groups = np.asarray(covered, dtype=bool), np.asarray(groups)
    return {g: float(covered[groups == g].mean()) for g in dict.fromkeys(groups.tolist())}


# --------------------------------------------------------------------------------------------
# VaR backtests
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest test class

    statistic: float
    p_value: float
    dof: int

    def passed(self, level: float = 0.05) -> bool:
        return self.p_value > level


def kupiec_pof(breaches: np.ndarray, alpha: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: breach probability = α. With x breaches in n days and p̂ = x/n,

        LR_pof = −2 [ (n−x) ln(1−α) + x ln α − (n−x) ln(1−p̂) − x ln p̂ ]  ~  χ²(1)
    """
    b = np.asarray(breaches, dtype=bool)
    n, x = b.size, int(b.sum())
    p_hat = x / n
    # xlogy(0, 0) = 0 handles the x = 0 and x = n edge cases.
    ll_null = xlogy(n - x, 1 - alpha) + xlogy(x, alpha)
    ll_alt = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)), 1)


def christoffersen_independence(breaches: np.ndarray) -> TestResult:
    """Christoffersen (1998) independence test: are breaches clustered?

    With n_ij = #transitions from state i to state j (1 = breach),
    π01 = n01/(n00+n01), π11 = n11/(n10+n11), π = (n01+n11)/(n00+n01+n10+n11):

        LR_ind = −2 [ ln L(π) − ln L(π01, π11) ]  ~  χ²(1)
    """
    b = np.asarray(breaches, dtype=int)
    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    def _ratio(a: int, b_: int) -> float:
        return a / (a + b_) if a + b_ else 0.0

    pi01, pi11 = _ratio(n01, n00), _ratio(n11, n10)
    pi = _ratio(n01 + n11, n00 + n10)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return TestResult(lr, float(stats.chi2.sf(lr, 1)), 1)


def christoffersen_conditional(breaches: np.ndarray, alpha: float) -> TestResult:
    """Conditional coverage: LR_cc = LR_pof + LR_ind ~ χ²(2). Joint test of rate and clustering."""
    lr = kupiec_pof(breaches, alpha).statistic + christoffersen_independence(breaches).statistic
    return TestResult(lr, float(stats.chi2.sf(lr, 2)), 2)


# --------------------------------------------------------------------------------------------
# Comparing forecasts
# --------------------------------------------------------------------------------------------


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 0) -> TestResult:
    """Diebold–Mariano (1995) test that two forecasts have equal expected loss.

    With the loss differential d_t = loss_a − loss_b, the statistic is d̄ / sqrt(V̂ / n), where V̂ is
    the Newey–West (Bartlett) long-run variance of d_t using ``lag`` autocovariances. Use
    ``lag`` ≥ h − 1 for h-day-ahead forecasts, whose errors overlap. A negative statistic means
    forecast A has the lower loss. The p-value is two-sided, from the normal distribution.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * float(dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return TestResult(0.0, 1.0, n - 1)
    stat = float(d.mean() / np.sqrt(lrv / n))
    return TestResult(stat, float(2 * stats.norm.sf(abs(stat))), n - 1)
