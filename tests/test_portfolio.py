import numpy as np
import pandas as pd
import pytest

from aegis.portfolio import (
    StrategyResult,
    allocation_returns,
    calibrate_scale,
    erc_weights,
    inverse_vol_weights,
    max_drawdown,
    run_exposure,
    strategy_metrics,
    var_limit_exposure,
    vol_target_exposure,
)

IDX = pd.bdate_range("2021-01-01", periods=6)


def test_exposure_at_t_earns_return_of_t_plus_1_and_pays_costs():
    r = pd.Series([0.0, 0.01, -0.02, 0.03, 0.0, 0.01], index=IDX)
    expo = pd.Series([1.0, 2.0, 2.0, 0.5, 0.5, 0.5], index=IDX)
    res = run_exposure(r, expo, cost_bps=10)
    assert res.gross.iloc[0] == pytest.approx(1.0 * 0.01)
    assert res.gross.iloc[1] == pytest.approx(2.0 * -0.02)
    assert np.isnan(res.gross.iloc[-1])  # no t+1 return yet
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, 1.0, 0.0, 1.5, 0.0, 0.0])
    assert res.net.iloc[3] == pytest.approx(0.5 * 0.0 - 1.5 * 10 / 1e4)


def test_vol_target_and_var_limit_formulas():
    sig = pd.Series([0.01, 0.005, 0.001], index=IDX[:3])
    e = vol_target_exposure(sig, 0.10, scale=1.0, max_leverage=2.0)
    assert e.iloc[0] == pytest.approx(0.10 / (0.01 * np.sqrt(252)))
    assert e.iloc[2] == 2.0  # capped
    v = var_limit_exposure(pd.Series([-0.04, -0.01], index=IDX[:2]), 0.02, max_leverage=1.5)
    assert v.tolist() == [0.5, 1.5]


def test_calibrate_scale_hits_target_on_the_calibration_sample():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=800)
    sig = pd.Series(np.exp(rng.normal(np.log(0.01), 0.3, 800)), index=idx)
    r = pd.Series(rng.normal(0, 1, 800), index=idx) * sig.shift(1).bfill()
    k = calibrate_scale(r, sig, 0.10, max_leverage=2.0)
    res = run_exposure(r, vol_target_exposure(sig, 0.10, k, max_leverage=np.inf), cost_bps=0)
    assert res.gross.std() * np.sqrt(252) == pytest.approx(0.10, rel=1e-9)


def test_erc_equalizes_risk_contributions_and_reduces_to_inverse_vol():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 4))
    cov = (a @ a.T + 4 * np.eye(4))[None] * 1e-4
    w = erc_weights(cov)[0]
    rc = w * (cov[0] @ w)
    assert w.sum() == pytest.approx(1) and (w > 0).all()
    np.testing.assert_allclose(rc / rc.sum(), 0.25, atol=1e-8)
    diag = np.diag([0.04, 0.01, 0.0025])[None]
    np.testing.assert_allclose(erc_weights(diag)[0], inverse_vol_weights(diag)[0], atol=1e-10)
    assert np.isnan(erc_weights(np.full((1, 2, 2), np.nan))).all()


def test_allocation_returns_use_previous_day_weights():
    r = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, 0.0, 0.10]}, index=IDX[:3])
    w = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=IDX[:3])
    port, turn = allocation_returns(r, w)
    assert port.iloc[1] == pytest.approx(np.expm1(0.10))  # day-0 weights (all A) earn day-1
    assert port.iloc[2] == pytest.approx(np.expm1(0.10))  # day-1 weights (all B) earn day-2
    assert turn.iloc[1] == 2.0


def test_drawdown_and_metrics():
    r = pd.Series([0.10, -0.50, 0.20], index=IDX[:3])
    assert max_drawdown(r) == pytest.approx(-0.50)
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(2)
    net = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    sr = StrategyResult(pd.Series(1.0, index=idx), net, net, pd.Series(0.0, index=idx))
    m = strategy_metrics(sr, target_ann=0.10)
    assert m["ann_vol"] == pytest.approx(net.std() * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(m["ann_return"] / m["ann_vol"])
    assert 0 <= m["overshoot_share"] <= 1 and m["cost_drag"] == 0
