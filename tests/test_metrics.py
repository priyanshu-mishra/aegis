import math

import numpy as np
import pytest

from aegis.metrics import (
    block_bootstrap_ci,
    christoffersen_conditional,
    christoffersen_independence,
    coverage,
    coverage_by_group,
    diebold_mariano,
    interval_score,
    is_covered,
    kupiec_pof,
    mean_width,
    paired_block_bootstrap_ci,
    rolling_coverage,
)


def test_coverage_and_width_basic():
    y = np.array([0.0, 2.0, 5.0])
    lo, hi = np.array([-1.0, 0.0, 6.0]), np.array([1.0, 1.0, 4.0])  # last is empty (lo > hi)
    assert is_covered(y, lo, hi).tolist() == [True, False, False]
    assert coverage(y, lo, hi) == pytest.approx(1 / 3)
    assert mean_width(lo, hi) == pytest.approx((2 + 1 + 0) / 3)


def test_interval_score_hand_computed():
    y = np.array([0.0, 3.0, -2.0])
    lo, hi = np.full(3, -1.0), np.full(3, 1.0)
    # widths 2; penalties (2/0.1)·{0, 2, 1} = {0, 40, 20}
    assert interval_score(y, lo, hi, alpha=0.1) == pytest.approx((2 + 42 + 22) / 3)


def test_rolling_coverage_matches_naive_loop():
    c = np.random.default_rng(0).random(200) < 0.9
    w = 20
    out = rolling_coverage(c, w)
    assert np.all(np.isnan(out[: w - 1]))
    naive = [c[i - w + 1 : i + 1].mean() for i in range(w - 1, len(c))]
    np.testing.assert_allclose(out[w - 1 :], naive)
    assert np.all(np.isnan(rolling_coverage(c[:5], w)))


def test_coverage_by_group():
    got = coverage_by_group([True, False, True, True], ["low", "low", "high", "high"])
    assert got == {"low": 0.5, "high": 1.0}


# ---------------------------------------------------------------------------------------------
# Kupiec
# ---------------------------------------------------------------------------------------------


def test_kupiec_matches_textbook_formula():
    n, x, a = 250, 5, 0.01
    b = np.zeros(n, bool)
    b[:x] = True
    p = x / n
    expected = -2 * (
        (n - x) * math.log(1 - a) + x * math.log(a) - (n - x) * math.log(1 - p) - x * math.log(p)
    )
    res = kupiec_pof(b, a)
    assert res.statistic == pytest.approx(expected)
    assert res.p_value == pytest.approx(math.erfc(math.sqrt(expected / 2)))  # χ²(1) survival


def test_kupiec_edge_cases():
    zero = kupiec_pof(np.zeros(250, bool), 0.01)
    assert zero.statistic == pytest.approx(-2 * 250 * math.log(0.99))
    exact = np.zeros(100, bool)
    exact[:5] = True
    res = kupiec_pof(exact, 0.05)
    assert res.statistic == pytest.approx(0, abs=1e-12) and res.p_value == pytest.approx(1)
    assert kupiec_pof(np.ones(50, bool), 0.05).p_value < 1e-10


# ---------------------------------------------------------------------------------------------
# Christoffersen
# ---------------------------------------------------------------------------------------------


def test_christoffersen_matches_hand_computation():
    b = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    n00, n01, n10, n11 = 7, 1, 1, 2
    pi01, pi11 = n01 / (n00 + n01), n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    ll_alt = (
        n00 * math.log(1 - pi01)
        + n01 * math.log(pi01)
        + n10 * math.log(1 - pi11)
        + n11 * math.log(pi11)
    )
    assert christoffersen_independence(b).statistic == pytest.approx(-2 * (ll_null - ll_alt))


def test_christoffersen_size_on_iid_breaches():
    # Under H0 a 5%-level test should reject about 5% of iid sequences, not 0% and not 30%.
    rng = np.random.default_rng(0)
    rejections = [
        not christoffersen_independence(rng.random(1000) < 0.05).passed() for _ in range(400)
    ]
    assert 0.01 <= np.mean(rejections) <= 0.10


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(0)
    clustered = np.zeros(2000, bool)
    for start in rng.choice(1990, 20, replace=False):
        clustered[start : start + 5] = True  # breaches arrive in bursts
    assert not christoffersen_independence(clustered).passed()


def test_christoffersen_conditional_is_sum():
    b = np.random.default_rng(1).random(500) < 0.07
    cc = christoffersen_conditional(b, 0.05)
    total = kupiec_pof(b, 0.05).statistic + christoffersen_independence(b).statistic
    assert cc.statistic == pytest.approx(total)
    assert cc.dof == 2


def test_christoffersen_no_breaches_is_zero():
    assert christoffersen_independence(np.zeros(100, bool)).statistic == 0


def test_diebold_mariano_detects_a_better_forecast_and_ignores_a_tie():
    rng = np.random.default_rng(0)
    good = rng.normal(0, 1, 2000) ** 2
    worse = good + 0.3 + rng.normal(0, 0.2, 2000)
    res = diebold_mariano(good, worse, lag=4)
    assert res.statistic < 0 and res.p_value < 1e-6
    same = diebold_mariano(good, good + rng.normal(0, 0.2, 2000), lag=4)
    assert same.p_value > 0.01
    assert diebold_mariano(good, good).p_value == 1.0  # identical forecasts


def test_diebold_mariano_matches_hand_calculation_without_lags():
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 2.0, 2.0, 3.0])
    d = a - b
    expected = d.mean() / np.sqrt(np.var(d) / len(d))  # lag 0: the plain t-type statistic
    assert diebold_mariano(a, b).statistic == pytest.approx(expected)


def test_diebold_mariano_long_run_variance_grows_with_positive_autocorrelation():
    rng = np.random.default_rng(1)
    e = rng.normal(size=3000)
    d = np.convolve(e, np.ones(5) / 5, mode="same") + 0.05  # a moving average is autocorrelated
    naive, robust = diebold_mariano(d, 0 * d, lag=0), diebold_mariano(d, 0 * d, lag=8)
    assert abs(robust.statistic) < abs(naive.statistic)


def test_block_bootstrap_ci_covers_the_truth_at_about_its_level():
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(200):
        x = (rng.random(1500) < 0.05).astype(float)
        lo, hi = block_bootstrap_ci(x, block=10, n_boot=400, seed=trial)
        hits += lo <= 0.05 <= hi
    assert 0.88 <= hits / 200 <= 0.99


def test_block_bootstrap_ci_is_wider_when_flags_cluster():
    rng = np.random.default_rng(1)
    iid = (rng.random(2000) < 0.05).astype(float)
    runs = np.repeat((rng.random(100) < 0.05).astype(float), 20)  # same rate, in 20-day bursts
    w = lambda x: np.subtract(*block_bootstrap_ci(x, block=21, seed=0)[::-1])  # noqa: E731
    assert w(runs) > 2 * w(iid)


def test_block_bootstrap_ci_is_reproducible_and_brackets_the_mean():
    x = np.random.default_rng(2).random(500) < 0.3
    a, b = block_bootstrap_ci(x, seed=7), block_bootstrap_ci(x, seed=7)
    assert a == b and a[0] < x.mean() < a[1]


def test_paired_ci_excludes_zero_for_a_real_difference_and_includes_it_for_a_tie():
    rng = np.random.default_rng(3)
    base = (rng.random(2000) < 0.05).astype(float)
    worse = np.maximum(base, (rng.random(2000) < 0.04).astype(float))  # about 4 points more
    lo, hi = paired_block_bootstrap_ci(worse, base, seed=0)
    assert lo > 0
    lo, hi = paired_block_bootstrap_ci(base, base.copy(), seed=0)
    assert lo == hi == 0.0
