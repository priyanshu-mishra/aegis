"""Statistical and behavioural tests for the conformal layer.

The coverage tests check the finite-sample guarantee (eq. 2 in conformal_ncm):
on exchangeable data, E[coverage] = k/(n+1) with k = ⌈(n+1)(1−α)⌉. With n = 99, α = 0.05 that
is exactly 0.95. They use deliberately *misspecified* point models: the guarantee must hold no
matter how bad the model is; only the interval width should suffer.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from aegis.config import ConformalConfig
from aegis.conformal_ncm import (
    AbsoluteResidual,
    AdaptiveConformal,
    Forecast,
    LowerTailResidual,
    NormalizedResidual,
    QuantileResidual,
    RollingConformal,
    SplitConformal,
    build_method,
    conformal_quantile,
    run_online,
)
from aegis.metrics import coverage

ALPHA = 0.05

# ---------------------------------------------------------------------------------------------
# conformal_quantile
# ---------------------------------------------------------------------------------------------


def _bruteforce_quantile(scores, alpha_milli):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - Fraction(alpha_milli, 1000)))  # exact rational arithmetic
    return math.inf if k > n else sorted(scores)[k - 1]


@given(
    scores=st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=300),
    alpha_milli=st.integers(1, 999),
)
def test_quantile_matches_bruteforce(scores, alpha_milli):
    assert conformal_quantile(np.array(scores), alpha_milli / 1000) == _bruteforce_quantile(
        scores, alpha_milli
    )


def test_quantile_float_edge_cases():
    s = np.arange(1, 20, dtype=float)  # n = 19 → k = ⌈20 · 0.95⌉ = 19 exactly
    assert conformal_quantile(s, 0.05) == 19.0
    assert conformal_quantile(s[:18], 0.05) == math.inf  # n = 18 → k = 19 > n
    assert conformal_quantile(s, 0.0) == math.inf
    assert conformal_quantile(s, -0.1) == math.inf
    assert conformal_quantile(s, 1.0) == -math.inf
    with pytest.raises(ValueError):
        conformal_quantile(np.array([]), 0.1)


# ---------------------------------------------------------------------------------------------
# Finite-sample coverage on exchangeable data
# ---------------------------------------------------------------------------------------------


def _hetero_data(rng, n):
    """Heteroskedastic, heavy-tailed data and a biased, misspecified forecast."""
    x = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x) + (0.1 + x) * rng.standard_t(4, n)
    mu = 0.8 * np.sin(2 * np.pi * x)
    sigma = 0.5 + 0.5 * x
    return Forecast(mu=mu, sigma=sigma, lo=mu - sigma, hi=mu + sigma), y


def _mean_coverage(make_method, trials=2000, n_cal=99, n_test=50, seed=0, one_sided=False):
    rng = np.random.default_rng(seed)
    covs = np.empty(trials)
    for i in range(trials):
        f, y = _hetero_data(rng, n_cal + n_test)
        m = make_method().fit(f[:n_cal], y[:n_cal])
        lo, hi = m.predict(f[n_cal:])
        covs[i] = coverage(y[n_cal:], lo, hi)
    return covs.mean()


@pytest.mark.parametrize(
    "ncm", [AbsoluteResidual, NormalizedResidual, QuantileResidual, LowerTailResidual]
)
def test_split_conformal_finite_sample_coverage(ncm):
    n_cal = 99
    mean_cov = _mean_coverage(lambda: SplitConformal(ncm(), ALPHA), n_cal=n_cal)
    # Theory: 1−α ≤ E[cov] ≤ 1−α + 1/(n+1). Tolerance ≈ 4 Monte-Carlo standard errors.
    tol = 0.0035
    assert 1 - ALPHA - tol <= mean_cov <= 1 - ALPHA + 1 / (n_cal + 1) + tol


def test_undersized_calibration_set_gives_infinite_interval():
    rng = np.random.default_rng(1)
    f, y = _hetero_data(rng, 10)
    m = SplitConformal(AbsoluteResidual(), ALPHA).fit(f, y)  # k = ⌈11·0.95⌉ = 11 > 10
    lo, hi = m.predict(f)
    assert np.all(np.isinf(lo)) and np.all(np.isinf(hi))


def test_normalized_intervals_adapt_to_sigma():
    rng = np.random.default_rng(2)
    f, y = _hetero_data(rng, 500)
    lo, hi = SplitConformal(NormalizedResidual(), ALPHA).fit(f, y).predict(f)
    width = hi - lo
    assert np.allclose(width / f.sigma, width[0] / f.sigma[0])  # width ∝ σ̂
    lo_a, hi_a = SplitConformal(AbsoluteResidual(), ALPHA).fit(f, y).predict(f)
    assert np.ptp(hi_a - lo_a) == pytest.approx(0)  # absolute residual: constant width


# ---------------------------------------------------------------------------------------------
# Distribution shift: split breaks, ACI and rolling recover
# ---------------------------------------------------------------------------------------------


def _shift_stream(seed=3, n_cal=500, n=4000, shift_at=2000, jump=3.0):
    """Model believes σ = 1 forever; true σ triples halfway through the test stream."""
    rng = np.random.default_rng(seed)
    sig = np.concatenate([np.ones(n_cal + shift_at), np.full(n - shift_at, jump)])
    y = sig * rng.standard_normal(n_cal + n)
    f = Forecast(mu=np.zeros(n_cal + n), sigma=np.ones(n_cal + n))
    return f[:n_cal], y[:n_cal], f[n_cal:], y[n_cal:]


def test_split_conformal_undercovers_after_shift():
    f_cal, y_cal, f, y = _shift_stream()
    res = run_online(SplitConformal(NormalizedResidual(), ALPHA).fit(f_cal, y_cal), f, y)
    assert res.covered[:2000].mean() > 0.93
    assert res.covered[2000:].mean() < 0.6  # |3ε| ≤ 1.96 only ~48% of the time


def test_aci_long_run_bound_holds_under_shift():
    gamma = 0.01
    f_cal, y_cal, f, y = _shift_stream()
    m = AdaptiveConformal(NormalizedResidual(), ALPHA, gamma=gamma).fit(f_cal, y_cal)
    res = run_online(m, f, y, feedback_lag=1)
    T = len(y)
    miss_rate = 1 - res.covered.mean()
    bound = (max(ALPHA, 1 - ALPHA) + gamma) / (gamma * T)  # eq. (4), deterministic
    assert abs(miss_rate - ALPHA) <= bound
    # α_t must have dropped after the shift to widen the intervals.
    assert res.alpha_path[-1] < ALPHA


def test_aci_with_feedback_lag_still_tracks_target():
    f_cal, y_cal, f, y = _shift_stream()
    m = AdaptiveConformal(NormalizedResidual(), ALPHA, gamma=0.01).fit(f_cal, y_cal)
    res = run_online(m, f, y, feedback_lag=5)
    assert abs(res.covered.mean() - (1 - ALPHA)) < 0.03


def test_rolling_conformal_recovers_after_window():
    f_cal, y_cal, f, y = _shift_stream()
    m = RollingConformal(NormalizedResidual(), ALPHA, window=500).fit(f_cal, y_cal)
    res = run_online(m, f, y)
    assert res.covered[-1000:].mean() > 0.93


# ---------------------------------------------------------------------------------------------
# Walk-forward mechanics
# ---------------------------------------------------------------------------------------------


class _SpySplit(SplitConformal):
    """Records which outcome indices were revealed at each prediction step."""

    def fit(self, f_cal, y_cal):
        self.revealed, self.log = [], []
        return super().fit(f_cal, y_cal)

    def update(self, f, y, covered):
        self.revealed.append(int(y[0]))

    def predict(self, f):
        self.log.append(max(self.revealed, default=-1))
        return super().predict(f)


@pytest.mark.parametrize("lag", [1, 5])
def test_run_online_never_reveals_outcomes_early(lag):
    n = 40
    y = np.arange(n, dtype=float)  # outcome value == its index
    f = Forecast(mu=np.zeros(n))
    m = _SpySplit(AbsoluteResidual(), ALPHA).fit(f, y)
    run_online(m, f, y, feedback_lag=lag)
    for t, latest in enumerate(m.log):
        assert latest <= max(t - lag, -1)  # -1: nothing revealed yet
    assert m.revealed == list(range(n - lag))


def test_run_online_rejects_zero_lag():
    f = Forecast(mu=np.zeros(5))
    with pytest.raises(ValueError):
        run_online(SplitConformal(AbsoluteResidual(), ALPHA).fit(f, np.zeros(5)), f, np.zeros(5), 0)


def test_split_online_equals_batch():
    rng = np.random.default_rng(4)
    f, y = _hetero_data(rng, 300)
    m = SplitConformal(QuantileResidual(), ALPHA).fit(f[:100], y[:100])
    lo, hi = m.predict(f[100:])
    res = run_online(m, f[100:], y[100:])
    np.testing.assert_allclose(res.lo, lo)
    np.testing.assert_allclose(res.hi, hi)


# ---------------------------------------------------------------------------------------------
# Forecast container and factory
# ---------------------------------------------------------------------------------------------


def test_forecast_validation_and_slicing():
    with pytest.raises(ValueError):
        Forecast(mu=np.zeros(3), sigma=np.ones(4))
    with pytest.raises(ValueError):
        Forecast(mu=np.zeros(2), sigma=np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        Forecast()
    f = Forecast(mu=np.arange(5), sigma=np.ones(5))
    assert len(f[1:3]) == 2 and f[1:3].lo is None
    with pytest.raises(ValueError, match="sigma"):
        SplitConformal(NormalizedResidual(), ALPHA).fit(Forecast(mu=np.zeros(30)), np.zeros(30))


def test_build_method_covers_every_config_name():
    for name in ConformalConfig().methods:
        assert build_method(name, ALPHA).alpha == ALPHA
    assert isinstance(build_method("aci", ALPHA, base_ncm="cqr").ncm, QuantileResidual)
    with pytest.raises(ValueError):
        build_method("nope", ALPHA)
    with pytest.raises(ValueError):
        build_method("split_absolute", 1.2)
