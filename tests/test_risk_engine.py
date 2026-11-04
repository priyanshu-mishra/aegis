import numpy as np
import pandas as pd
import pytest

from aegis.baselines import garch_var, gaussian_interval, gaussian_var, historical_var
from aegis.data.prices import build_market
from aegis.data.synthetic import FACTORS, generate_world
from aegis.features import build_features, build_targets
from aegis.risk_engine import (
    ShockOverlay,
    VolForecaster,
    apply_shock_overlay,
    ewma_covariance,
    fit_shock_overlay,
    portfolio_vol,
    purged_kfold,
    return_forecast,
)

TICKERS = ["SPY", "TLT", "GLD", "USO"]


@pytest.fixture(scope="module")
def world():
    return generate_world(TICKERS, n_days=2500, seed=21)


@pytest.fixture(scope="module")
def market(world):
    return build_market(world.prices, world.features, "equal", None, "synthetic")


@pytest.fixture(scope="module")
def xy(market):
    X = build_features(market)
    Y = build_targets(market, horizon=5)
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return X[ok], Y[ok]


# ---------------------------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------------------------


def test_ewma_recursion_matches_manual():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (60, 2)))
    lam, burn = 0.9, 10
    cov = ewma_covariance(r, lam, burn_in=burn)
    assert np.isnan(cov[: burn - 1]).all()
    x = r.to_numpy()
    s = x[:burn].T @ x[:burn] / burn
    for t in range(burn, 60):
        s = lam * s + (1 - lam) * np.outer(x[t], x[t])
    np.testing.assert_allclose(cov[-1], s)
    assert all(np.linalg.eigvalsh(c).min() >= 0 for c in cov[burn - 1 :])


def test_ewma_ignores_future():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (100, 3)))
    r2 = r.copy()
    r2.iloc[51:] *= 10
    np.testing.assert_array_equal(ewma_covariance(r)[:51], ewma_covariance(r2)[:51])


def test_portfolio_vol_hand_computed():
    cov = np.array([[[0.04, 0.01], [0.01, 0.09]]])
    w = np.array([[0.5, 0.5]])
    expected = np.sqrt(0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.01)
    assert portfolio_vol(cov, w)[0] == pytest.approx(expected)


def test_overlay_identity_and_psd():
    rng = np.random.default_rng(2)
    r = pd.DataFrame(rng.normal(0, 0.01, (80, 3)))
    cov = ewma_covariance(r)
    s = rng.normal(size=(80, 2))
    np.testing.assert_array_equal(apply_shock_overlay(cov, s, ShockOverlay.identity(3, 2)), cov)
    adj = apply_shock_overlay(
        cov, s, ShockOverlay(rng.normal(size=(3, 2)), rng.uniform(0.3, 1.2, 3), rng.normal(size=3))
    )
    assert all(np.linalg.eigvalsh(c).min() >= -1e-15 for c in adj[30:])
    # Correlations are unchanged by a diagonal rescaling.
    c0, c1 = cov[50], adj[50]
    corr = lambda c: c / np.sqrt(np.outer(np.diag(c), np.diag(c)))  # noqa: E731
    np.testing.assert_allclose(corr(c0), corr(c1))


def test_overlay_with_oracle_signals_improves_vol_forecast():
    """With the true latent shocks as signals, the overlay must beat the same overlay fit
    without signals (calibrated EWMA) out of sample. This proves the plumbing; M5 replaces
    the oracle with signals parsed from headlines.
    """
    world = generate_world(TICKERS, n_days=4000, seed=21)
    market = build_market(world.prices, world.features, "equal", None, "synthetic")
    r = market.returns
    cov = ewma_covariance(r)
    level = world.latent.abs().reindex(r.index)
    fresh = (level - level.shift(5)).clip(lower=0).fillna(0).add_suffix("_fresh")
    signals = level.join(fresh)
    n = len(r)
    train = np.arange(n) < n // 2

    full = fit_shock_overlay(cov, r, signals, train, horizon=5)
    base = fit_shock_overlay(cov, r, signals * 0, train, horizon=5)  # calibrated EWMA, no text
    assert full.B.shape == (4, 2 * len(FACTORS))
    assert np.allclose(base.B, 0)
    assert np.all((base.beta > 0.2) & (base.beta < 1))  # EWMA shrunk toward the mean

    fwd = np.log(np.maximum(np.sqrt((r**2).rolling(5).mean().shift(-5)), 1e-4)).to_numpy()

    def rmse(overlay):
        adj = apply_shock_overlay(cov, signals.to_numpy(), overlay)
        pred = np.log(np.sqrt(np.diagonal(adj, axis1=1, axis2=2)))
        return np.sqrt(np.nanmean(((fwd - pred)[~train]) ** 2))

    assert rmse(full) < 0.97 * rmse(base)


# ---------------------------------------------------------------------------------------------
# Features and targets
# ---------------------------------------------------------------------------------------------


def test_feature_columns_and_targets_alignment(market):
    X = build_features(market)
    assert {"log_rv_d", "log_rv_w", "log_rv_m", "log_ewma_vol", "log_vix", "dlog_vix_5"} <= set(
        X.columns
    )
    Y = build_targets(market, horizon=5)
    r = market.portfolio
    t = 100
    expected = np.log(np.sqrt((r.iloc[t + 1 : t + 6] ** 2).mean()))
    assert Y["log_vol_fwd"].iloc[t] == pytest.approx(expected)
    assert Y["ret_fwd"].iloc[t] == r.iloc[t + 1]
    assert Y["log_vol_fwd"].iloc[-5:].isna().all() and np.isnan(Y["ret_fwd"].iloc[-1])


def test_features_ignore_future(world):
    t = 300
    prices2, feats2 = world.prices.copy(), world.features.copy()
    prices2.iloc[t + 2 :] *= np.exp(np.linspace(0, 1, len(prices2) - t - 2))[:, None]
    feats2.iloc[t + 2 :] *= 3
    a = build_features(build_market(world.prices, world.features, "inverse_vol", None, "s"))
    b = build_features(build_market(prices2, feats2, "inverse_vol", None, "s"))
    # prices row t+1 → return row t (returns drop the first price row)
    pd.testing.assert_frame_equal(a.iloc[: t + 1], b.iloc[: t + 1])


def test_signals_are_joined_with_prefix(market):
    sig = pd.DataFrame({"sentiment": 0.1}, index=market.returns.index)
    assert "sig_sentiment" in build_features(market, signals=sig).columns


# ---------------------------------------------------------------------------------------------
# Point models
# ---------------------------------------------------------------------------------------------


def test_purged_kfold_respects_gap():
    folds = list(purged_kfold(100, 5, gap=5))
    assert len(folds) == 5
    covered = np.concatenate([te for _, te in folds])
    assert sorted(covered.tolist()) == list(range(100))
    for tr, te in folds:
        assert np.min(np.abs(tr[:, None] - te[None, :])) > 5


@pytest.mark.parametrize("kind", ["har_ols", "ridge", "lightgbm", "mlp"])
def test_vol_forecaster_beats_naive(kind, xy):
    X, Y = xy
    y = Y["log_vol_fwd"]
    cut = len(X) * 2 // 3
    m = VolForecaster(kind=kind, alpha=0.1).fit(X.iloc[:cut], y.iloc[:cut])
    f = m.predict(X.iloc[cut:])
    y_te = y.iloc[cut:].to_numpy()
    rmse = np.sqrt(np.mean((y_te - f.mu) ** 2))
    naive = np.sqrt(np.mean((y_te - y.iloc[:cut].mean()) ** 2))
    assert rmse < 0.9 * naive
    assert np.all(f.sigma > 0) and np.all(f.lo <= f.hi)
    # Raw quantile band is not guaranteed, but should be in the right neighbourhood.
    raw_cov = np.mean((y_te >= f.lo) & (y_te <= f.hi))
    assert 0.6 < raw_cov < 0.99
    assert m.oof_rmse_ > 0


def test_har_uses_only_har_columns(xy):
    X, Y = xy
    m = VolForecaster(kind="har_ols").fit(X.iloc[:500], Y["log_vol_fwd"].iloc[:500])
    assert m.columns_ == ["log_rv_d", "log_rv_w", "log_rv_m"]


def test_vol_forecaster_rejects_nan(xy):
    X, Y = xy
    X2 = X.iloc[:300].copy()
    X2.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        VolForecaster(kind="ridge").fit(X2, Y["log_vol_fwd"].iloc[:300])


def test_return_forecast():
    f = return_forecast(pd.Series(np.log([0.01, 0.02])))
    np.testing.assert_allclose(f.sigma, [0.01, 0.02])
    np.testing.assert_allclose(f.mu, 0)


# ---------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------


def test_gaussian_var_breach_rate_on_gaussian_data():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=20000)
    sigma = pd.Series(0.01, index=idx)
    r_next = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    lvl = gaussian_var(sigma, 0.05)
    assert lvl.iloc[0] == pytest.approx(-1.6448536 * 0.01, rel=1e-6)
    assert abs((r_next < lvl).mean() - 0.05) < 0.005


def test_historical_var_uses_past_only():
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(np.random.default_rng(4).normal(0, 0.01, 400), index=idx)
    r2 = r.copy()
    r2.iloc[301:] -= 0.5
    a, b = historical_var(r, 0.05), historical_var(r2, 0.05)
    pd.testing.assert_series_equal(a.iloc[:301], b.iloc[:301])
    assert a.iloc[:249].isna().all()
    assert a.iloc[299] == pytest.approx(np.quantile(r.iloc[50:300], 0.05))


def test_garch_var_no_lookahead_and_sane_breaches():
    idx = pd.bdate_range("2020-01-01", periods=800)
    rng = np.random.default_rng(5)
    r = pd.Series(rng.standard_t(5, 800) * 0.01, index=idx)
    a = garch_var(r, 0.05, idx[499])
    r2 = r.copy()
    r2.iloc[601:] *= 5
    b = garch_var(r2, 0.05, idx[499])
    pd.testing.assert_series_equal(a.iloc[:601], b.iloc[:601])
    breach = (r.shift(-1) < a).iloc[500:-1].mean()
    assert 0.02 < breach < 0.09


def test_gaussian_interval():
    lo, hi = gaussian_interval(np.zeros(2), 1.0, 0.05)
    np.testing.assert_allclose(hi, 1.959964, rtol=1e-6)
    np.testing.assert_allclose(lo, -hi)
