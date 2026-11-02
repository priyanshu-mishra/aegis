import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from aegis.baselines import garch_forecast, historical_var_horizon
from aegis.conformal_ncm import (
    AdaptiveConformal,
    Forecast,
    LowerTailResidual,
    SplitConformal,
    run_online,
)
from aegis.risk_measures import (
    acerbi_szekely_z2,
    basel_multiplier,
    basel_zone,
    gaussian_es,
    historical_es,
    market_risk_capital,
    risk_contributions,
    student_t_es_factor,
    traffic_light,
)

# ---------------------------------------------------------------------------------------------
# Expected Shortfall formulas
# ---------------------------------------------------------------------------------------------


def test_gaussian_es_matches_numerical_tail_mean():
    a, s = 0.025, 0.02
    q = stats.norm.ppf(a, scale=s)
    tail, _ = integrate.quad(lambda x: x * stats.norm.pdf(x, scale=s), -np.inf, q)
    expected = tail / a
    es = gaussian_es(pd.Series([s]), a).iloc[0]
    assert es == pytest.approx(expected, rel=1e-6)
    assert es < q  # ES lies beyond VaR


def test_student_t_es_factor_monte_carlo_and_normal_limit():
    rng = np.random.default_rng(0)
    nu, a = 5.0, 0.025
    x = rng.standard_t(nu, 4_000_000) * np.sqrt((nu - 2) / nu)  # unit variance
    q = np.quantile(x, a)
    mc = -x[x <= q].mean()
    assert student_t_es_factor(a, nu) == pytest.approx(mc, rel=0.01)
    normal = stats.norm.pdf(stats.norm.ppf(a)) / a
    assert student_t_es_factor(a, 1e6) == pytest.approx(normal, rel=1e-4)


def test_historical_es_uses_past_only_and_matches_manual():
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400), index=idx)
    es = historical_es(r, 0.05, window=250)
    w = r.iloc[50:300].to_numpy()
    q = np.quantile(w, 0.05)
    assert es.iloc[299] == pytest.approx(w[w <= q].mean())
    r2 = r.copy()
    r2.iloc[301:] -= 1
    pd.testing.assert_series_equal(es.iloc[:301], historical_es(r2, 0.05).iloc[:301])


# ---------------------------------------------------------------------------------------------
# Acerbi–Székely Z2
# ---------------------------------------------------------------------------------------------


def test_z2_is_zero_on_average_for_a_correct_model_and_negative_when_es_too_small():
    rng = np.random.default_rng(2)
    a, n = 0.025, 2000
    v = np.full(n, stats.norm.ppf(a))
    e = np.full(n, -stats.norm.pdf(stats.norm.ppf(a)) / a)
    z_ok = [acerbi_szekely_z2(rng.standard_normal(n), v, e, a).z2 for _ in range(300)]
    assert abs(np.mean(z_ok)) < 0.05
    # Fatter realized tails than forecast → ES underestimated → clearly negative Z2.
    z_bad = acerbi_szekely_z2(1.6 * rng.standard_normal(n), v, e, a)
    assert z_bad.z2 < -0.7 and z_bad.zone in ("yellow", "red")


def test_z2_hand_computed():
    x = np.array([-0.05, 0.01, -0.02, 0.0])
    v = np.full(4, -0.03)
    e = np.full(4, -0.04)
    # one breach: X = −0.05; Z2 = (−0.05 / 0.04) / (4 · 0.25) + 1 = −0.25
    assert acerbi_szekely_z2(x, v, e, 0.25).z2 == pytest.approx(-0.25)


# ---------------------------------------------------------------------------------------------
# Basel traffic light and capital
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k, zone, mult",
    [(0, "green", 3.0), (4, "green", 3.0), (5, "yellow", 3.4), (7, "yellow", 3.65),
     (9, "yellow", 3.85), (10, "red", 4.0), (25, "red", 4.0)],
)  # fmt: skip
def test_basel_table(k, zone, mult):
    assert basel_zone(k) == zone
    assert basel_multiplier(k) == pytest.approx(mult)


def test_traffic_light_counts_only_known_outcomes():
    idx = pd.bdate_range("2020-01-01", periods=30)
    breach = pd.Series(False, index=idx)
    breach.iloc[[3, 10, 11]] = True  # forecasts made on these days were breached the next day
    tl = traffic_light(breach, window=10)
    assert tl["exceptions"].iloc[:10].isna().all()  # window not yet full
    # On day 12 the outcomes of forecasts made on days 3..11 are known (shifted by one).
    assert tl["exceptions"].iloc[12] == 3
    assert tl["exceptions"].iloc[11] == 2  # day 11's own breach is not known until day 12
    assert tl["zone"].iloc[12] == "green"


def test_market_risk_capital_formula():
    idx = pd.bdate_range("2020-01-01", periods=5)
    var10 = pd.Series([-0.05, -0.05, -0.05, -0.05, -0.20], index=idx)
    mult = pd.Series(3.0, index=idx)
    cap = market_risk_capital(var10, mult, avg_window=4)
    assert cap.iloc[3] == pytest.approx(3 * 0.05)  # multiplier × average dominates
    assert cap.iloc[4] == pytest.approx(max(0.20, 3 * np.mean([0.05, 0.05, 0.05, 0.20])))


# ---------------------------------------------------------------------------------------------
# Risk decomposition
# ---------------------------------------------------------------------------------------------


def test_risk_contributions_sum_to_one_and_match_hand_calc():
    cov = np.array([[[0.04, 0.006], [0.006, 0.01]]])
    w = np.array([[0.5, 0.5]])
    rc = risk_contributions(cov, w)[0]
    sw = cov[0] @ w[0]
    var_p = w[0] @ sw
    np.testing.assert_allclose(rc, w[0] * sw / var_p)
    assert rc.sum() == pytest.approx(1.0)
    hedge = risk_contributions(np.array([[[0.04, -0.015], [-0.015, 0.01]]]), w)[0]
    assert hedge.sum() == pytest.approx(1.0) and hedge.min() < 0  # a hedge has a negative share


# ---------------------------------------------------------------------------------------------
# Conformal tail means and the ACI floor
# ---------------------------------------------------------------------------------------------


def test_conformal_tail_mean_is_mean_beyond_quantile():
    rng = np.random.default_rng(3)
    y = rng.standard_t(4, 500) * 0.01
    f = Forecast(mu=np.zeros(500), sigma=np.full(500, 0.01))
    m = SplitConformal(LowerTailResidual(), 0.05).fit(f, y)
    s = (0 - y) / 0.01
    assert m.tail_mean() == pytest.approx(s[s >= m.q].mean())
    assert m.tail_mean() >= m.q


def test_aci_floor_keeps_var_finite_after_a_burst_of_misses():
    n_cal, n = 300, 400
    rng = np.random.default_rng(4)
    y_cal = rng.normal(0, 0.01, n_cal)
    y = rng.normal(0, 0.01, n)
    y[50:70] = -0.2  # a crash: 20 consecutive breaches
    f_cal = Forecast(mu=np.zeros(n_cal), sigma=np.full(n_cal, 0.01))
    f = Forecast(mu=np.zeros(n), sigma=np.full(n, 0.01))
    plain = run_online(AdaptiveConformal(LowerTailResidual(), 0.01, 0.005).fit(f_cal, y_cal), f, y)
    floored = run_online(
        AdaptiveConformal(LowerTailResidual(), 0.01, 0.005, alpha_floor=0.002).fit(f_cal, y_cal),
        f, y,
    )  # fmt: skip
    assert np.isinf(plain.lo).any()  # plain ACI's α_t drops below 0: an unbounded VaR
    assert np.isfinite(floored.lo).all()
    with pytest.raises(ValueError):
        AdaptiveConformal(LowerTailResidual(), 0.01, alpha_floor=0.05)


# ---------------------------------------------------------------------------------------------
# Multi-day baselines
# ---------------------------------------------------------------------------------------------


def test_garch_multi_horizon_no_lookahead_and_variance_grows():
    idx = pd.bdate_range("2020-01-01", periods=700)
    r = pd.Series(np.random.default_rng(5).standard_t(5, 700) * 0.01, index=idx)
    f1 = garch_forecast(r, idx[399], horizon=1)
    f10 = garch_forecast(r, idx[399], horizon=10)
    assert (f10["sigma"] > f1["sigma"]).all()
    assert f10["sigma"].iloc[-1] == pytest.approx(f1["sigma"].iloc[-1] * np.sqrt(10), rel=0.3)
    r2 = r.copy()
    r2.iloc[501:] *= 4
    pd.testing.assert_frame_equal(f10.iloc[:501], garch_forecast(r2, idx[399], 10).iloc[:501])


def test_historical_var_horizon_past_only():
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(np.random.default_rng(6).normal(0, 0.01, 400), index=idx)
    a = historical_var_horizon(r, 0.01, 10)
    r2 = r.copy()
    r2.iloc[301:] -= 0.5
    pd.testing.assert_series_equal(a.iloc[:301], historical_var_horizon(r2, 0.01, 10).iloc[:301])
    assert a.iloc[:258].isna().all()  # needs 10 days for a sum, then 250 sums
