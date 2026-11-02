"""DtACI, Mondrian conformal, FHS / GJR baselines and the sensitivity grid."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.backtest import run_backtest
from aegis.baselines import fhs, garch_forecast
from aegis.cli import main
from aegis.config import load_config
from aegis.conformal_ncm import (
    AbsoluteResidual,
    DtACI,
    Forecast,
    MondrianConformal,
    NormalizedResidual,
    SplitConformal,
    regime_groups,
    run_online,
)
from aegis.sensitivity import ALPHAS, conformal_grid

ROOT = Path(__file__).resolve().parents[1]


def _shift_stream(seed=3, n_cal=500, n=4000, shift_at=2000, jump=3.0):
    rng = np.random.default_rng(seed)
    sig = np.concatenate([np.ones(n_cal + shift_at), np.full(n - shift_at, jump)])
    y = sig * rng.standard_normal(n_cal + n)
    f = Forecast(mu=np.zeros(n_cal + n), sigma=np.ones(n_cal + n))
    return f[:n_cal], y[:n_cal], f[n_cal:], y[n_cal:]


# ---------------------------------------------------------------------------------------------
# DtACI
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [1, 5])
def test_dtaci_tracks_target_through_a_shift(lag):
    f_cal, y_cal, f, y = _shift_stream()
    m = DtACI(NormalizedResidual(), 0.05).fit(f_cal, y_cal)
    res = run_online(m, f, y, feedback_lag=lag)
    assert abs(res.covered.mean() - 0.95) < 0.01
    assert res.covered[2000:].mean() > 0.93  # recovers after the variance triples
    assert res.alpha_path is not None and res.alpha_path[2050] < 0.05  # widened after misses
    assert m.weights.sum() == pytest.approx(1.0)


def test_dtaci_scores_experts_on_issued_levels_with_overlapping_targets():
    f_cal, y_cal, f, y = _shift_stream()
    smooth = np.ones(5) / 5
    yc5, y5 = np.convolve(y_cal, smooth, "same"), np.convolve(y, smooth, "same")
    res = run_online(DtACI(NormalizedResidual(), 0.05).fit(f_cal, yc5), f, y5, feedback_lag=5)
    assert res.covered.mean() > 0.94  # 0.935 before issued-level scoring
    with pytest.raises(ValueError):
        DtACI(NormalizedResidual(), 0.05, stride=0)


# ---------------------------------------------------------------------------------------------
# Mondrian
# ---------------------------------------------------------------------------------------------


def test_mondrian_gives_group_conditional_coverage():
    rng = np.random.default_rng(0)
    n = 9000
    g = rng.integers(0, 3, n).astype(float)
    y = np.array([0.5, 1.0, 2.0])[g.astype(int)] * rng.standard_normal(n)
    f = Forecast(mu=np.zeros(n), group=g)
    cal, te = slice(0, 4500), slice(4500, n)
    lo, hi = SplitConformal(AbsoluteResidual(), 0.05).fit(f[cal], y[cal]).predict(f[te])
    pooled = [((y[te] >= lo) & (y[te] <= hi))[g[te] == k].mean() for k in range(3)]
    lo, hi = MondrianConformal(AbsoluteResidual(), 0.05).fit(f[cal], y[cal]).predict(f[te])
    cond = [((y[te] >= lo) & (y[te] <= hi))[g[te] == k].mean() for k in range(3)]
    assert pooled[2] < 0.92 and pooled[0] > 0.98  # pooled threshold: wrong in both tails
    assert all(abs(c - 0.95) < 0.02 for c in cond)
    with pytest.raises(ValueError, match="group"):
        MondrianConformal(AbsoluteResidual(), 0.05).fit(Forecast(mu=np.zeros(10)), np.zeros(10))


def test_regime_groups():
    np.testing.assert_array_equal(regime_groups([0.1, 0.5, 0.9], np.array([0.3, 0.7])), [0, 1, 2])


# ---------------------------------------------------------------------------------------------
# FHS and GJR
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tdist_returns():
    idx = pd.bdate_range("2015-01-01", periods=1600)
    return pd.Series(np.random.default_rng(7).standard_t(4, 1600) * 0.01, index=idx)


def test_fhs_breach_rate_es_and_no_lookahead(tdist_returns):
    r = tdist_returns
    end = r.index[699]
    f = fhs(r, 0.01, end)
    breach = (r.shift(-1) < f["var"]).iloc[700:-1].mean()
    assert 0.003 < breach < 0.025
    ok = f.dropna()
    assert (ok["es"] <= ok["var"] + 1e-12).all()
    r2 = r.copy()
    r2.iloc[1001:] *= 5
    pd.testing.assert_frame_equal(f.iloc[:1001], fhs(r2, 0.01, end).iloc[:1001])


def test_gjr_differs_from_garch_and_ignores_future(tdist_returns):
    r = tdist_returns
    end = r.index[699]
    g0, g1 = garch_forecast(r, end, o=0), garch_forecast(r, end, o=1)
    assert not np.allclose(g0["sigma"].dropna(), g1["sigma"].dropna())
    r2 = r.copy()
    r2.iloc[1001:] *= 5
    pd.testing.assert_frame_equal(g1.iloc[:1001], garch_forecast(r2, end, o=1).iloc[:1001])


# ---------------------------------------------------------------------------------------------
# Backtest integration and sensitivity
# ---------------------------------------------------------------------------------------------

SMALL = ["data.synthetic.n_days=1500", "splits.train_end=2012-12-31", "splits.cal_end=2013-12-31"]


@pytest.fixture(scope="module")
def result():
    return run_backtest(load_config(ROOT / "configs/synthetic.yaml", SMALL))


def test_backtest_includes_v3_methods(result):
    assert {"dtaci_lo", "mondrian_lo"} <= set(result.vol.columns)
    for df in result.var.values():
        assert {"dtaci_var", "fhs_var", "gjr_var"} <= set(df.columns)
    assert {"dtaci_es", "fhs_es", "gjr_es"} <= set(result.es.columns)
    assert {"dtaci", "fhs", "gjr"} <= set(result.var_long.columns)
    assert {"fhs_exceptions", "gjr_capital", "dtaci_multiplier"} <= set(result.basel.columns)
    assert {"y", "mu", "group"} <= set(result.vol_cal.columns)
    assert "dtaci_vol" in result.alpha_paths


def test_conformal_grid_on_saved_forecasts(result):
    grid = conformal_grid(result)
    assert set(grid) == {"alpha", "gamma", "window"}
    assert set(grid["alpha"]) == {str(a) for a in ALPHAS}
    for a, v in grid["alpha"].items():
        assert abs(v["aci"]["coverage"] - (1 - float(a))) < 0.05


def test_cli_sensitivity_quick(tmp_path):
    over = []
    for s in [*SMALL, f"report.output_dir={tmp_path}", "report.figures=false"]:
        over += ["--set", s]
    cfg = str(ROOT / "configs/synthetic.yaml")
    assert main(["backtest", "-c", cfg, *over]) == 0
    assert main(["sensitivity", "-c", cfg, *over, "--quick"]) == 0
    assert (tmp_path / "synthetic" / "sensitivity.md").exists()
