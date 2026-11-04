"""Regime-conditional ACI and the early-warning signal (extension B)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.backtest import run_backtest
from aegis.config import load_config
from aegis.conformal_ncm import (
    AbsoluteResidual,
    AdaptiveConformal,
    Forecast,
    LowerTailResidual,
    RegimeACI,
    run_online,
)
from aegis.data import load_dataset
from aegis.early_warning import STRESS_Z, StressModel, stress_features, stress_label

ROOT = Path(__file__).resolve().parents[1]


def _regime_data(n=3000, seed=0):
    """Two regimes with very different noise, so a pooled interval over- and under-covers."""
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < 0.3).astype(float)
    y = rng.normal(0, np.where(g == 1, 3.0, 1.0))
    return Forecast(mu=np.zeros(n), group=g), y


def test_regime_aci_with_one_regime_equals_plain_aci():
    f, y = _regime_data()
    f1 = Forecast(mu=f.mu, group=np.zeros(len(y)))
    cal, te = slice(0, 500), slice(500, 1500)
    a = AdaptiveConformal(AbsoluteResidual(), 0.1, gamma=0.01).fit(f1[cal], y[cal])
    b = RegimeACI(AbsoluteResidual(), 0.1, gamma=0.01).fit(f1[cal], y[cal])
    ra = run_online(a, f1[te], y[te], feedback_lag=1)
    rb = run_online(b, f1[te], y[te], feedback_lag=1)
    np.testing.assert_allclose(ra.lo, rb.lo)
    np.testing.assert_allclose(ra.hi, rb.hi)


def test_regime_aci_covers_each_regime_when_pooled_aci_does_not():
    f, y = _regime_data()
    cal, te = slice(0, 1000), slice(1000, 3000)
    res = run_online(
        RegimeACI(AbsoluteResidual(), 0.1, gamma=0.01).fit(f[cal], y[cal]), f[te], y[te], 1
    )
    g = f.group[te]
    for k in (0.0, 1.0):
        assert abs(res.covered[g == k].mean() - 0.9) < 0.04


def test_regime_aci_never_uses_an_outcome_before_it_is_revealed():
    f, y = _regime_data(1200)
    cal, te = slice(0, 400), slice(400, 1200)
    y2 = y.copy()
    y2[700:] += 50.0  # change outcomes from step 300 of the test period onward
    kw = dict(feedback_lag=5)
    a = run_online(RegimeACI(AbsoluteResidual(), 0.1).fit(f[cal], y[cal]), f[te], y[te], **kw)
    b = run_online(RegimeACI(AbsoluteResidual(), 0.1).fit(f[cal], y2[cal]), f[te], y2[te], **kw)
    np.testing.assert_allclose(a.lo[:305], b.lo[:305])  # intervals up to step 304 are unchanged


def test_regime_aci_needs_groups():
    f = Forecast(mu=np.zeros(10))
    with pytest.raises(ValueError):
        RegimeACI(AbsoluteResidual(), 0.1).fit(f, np.zeros(10))


def test_regime_aci_falls_back_to_pooled_for_an_unseen_regime():
    f, y = _regime_data(800)
    m = RegimeACI(LowerTailResidual(), 0.1).fit(
        Forecast(mu=f.mu[:400], sigma=np.ones(400), group=np.zeros(400)), y[:400]
    )
    lo, _ = m.predict(Forecast(mu=np.zeros(1), sigma=np.ones(1), group=np.array([2.0])))
    assert np.isfinite(lo[0])


@pytest.fixture(scope="module")
def synth():
    cfg = load_config(
        ROOT / "configs/synthetic.yaml",
        [
            "data.synthetic.n_days=2500",
            "splits.train_end=2012-12-31",
            "splits.cal_end=2014-06-30",
            "conformal.early_warning=true",
        ],
    )
    ds = load_dataset(cfg)
    return cfg, ds, run_backtest(cfg, dataset=ds)


def test_stress_features_use_only_the_past(synth):
    _, ds, _ = synth
    m = ds.market
    sigma = m.portfolio.rolling(21).std().bfill().clip(lower=1e-4)
    t = m.returns.index[1500]
    full = stress_features(m, sigma).loc[:t]
    cut = type(m)(
        prices=m.prices.loc[:t],
        returns=m.returns.loc[:t],
        features=m.features.loc[:t],
        weights=m.weights.loc[:t],
        portfolio=m.portfolio.loc[:t],
        source=m.source,
    )
    short = stress_features(cut, sigma.loc[:t])
    pd.testing.assert_frame_equal(full, short)


def test_stress_label_marks_windows_that_contain_a_bad_day():
    n = 20
    ret_next = pd.Series(np.zeros(n))
    ret_next[10] = STRESS_Z * 2  # day t+1 = 10 is a breach of a unit-sigma Gaussian VaR
    lab = stress_label(ret_next, pd.Series(np.ones(n)))
    assert list(lab.iloc[6:11]) == [1.0] * 5  # windows starting at t = 6 … 10 contain it
    assert lab.iloc[5] == 0.0 and lab.iloc[11] == 0.0
    assert lab.iloc[-4:].isna().all()


def test_stress_model_learns_a_planted_signal():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    y = pd.Series((X["a"] + 0.3 * rng.normal(size=2000) > 1).astype(float))
    p = StressModel().fit(X.iloc[:1500], y.iloc[:1500]).proba(X.iloc[1500:])
    assert p[y.iloc[1500:].to_numpy() == 1].mean() > p[y.iloc[1500:].to_numpy() == 0].mean() + 0.2


def test_early_warning_backtest_adds_methods_and_reports_auc(synth):
    cfg, _, res = synth
    for a in cfg.conformal.var_alphas:
        assert {"aci_regime_var", "aci_stress_var"} <= set(res.var[a].columns)
    ew = res.info["early_warning"]
    assert 0 < ew["base_rate_train"] < 1 and 0 <= ew["auc_cal"] <= 1


def test_early_warning_is_off_by_default():
    assert load_config(None).conformal.early_warning is False


def test_early_warning_backtest_has_no_lookahead():
    """Perturbing every price after test day t must not change a regime VaR level dated ≤ t."""
    from aegis.data import Dataset
    from aegis.data.prices import build_market
    from aegis.data.synthetic import generate_world

    cfg = load_config(
        ROOT / "configs/synthetic.yaml",
        [
            "data.synthetic.n_days=1500",
            "splits.train_end=2012-12-31",
            "splits.cal_end=2013-12-31",
            "conformal.early_warning=true",
        ],
    )
    world = generate_world(cfg.data.tickers, n_days=1500, seed=cfg.seed)

    def run(prices):
        market = build_market(prices, world.features, "equal", None, "synthetic")
        return run_backtest(cfg, Dataset(market, world.news))

    a = run(world.prices)
    t = a.vol.index[200]
    shocked = world.prices.copy()
    after = shocked.index > t
    shocked.loc[after] *= np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.05, after.sum())))[
        :, None
    ]
    b = run(shocked)
    for alpha in a.var:
        cols = ["aci_regime_var", "aci_stress_var"]
        pd.testing.assert_frame_equal(a.var[alpha].loc[:t, cols], b.var[alpha].loc[:t, cols])
    assert (
        not a.var[0.01]
        .loc[a.var[0.01].index > t, "aci_stress_var"]
        .equals(b.var[0.01].loc[b.var[0.01].index > t, "aci_stress_var"])
    )
