import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aegis.backtest import BacktestResult, make_splits, run_backtest
from aegis.cli import main
from aegis.config import load_config
from aegis.data import Dataset
from aegis.data.prices import build_market
from aegis.data.synthetic import generate_world
from aegis.report import write_report

ROOT = Path(__file__).resolve().parents[1]
SMALL = [
    "data.synthetic.n_days=1500",
    "splits.train_end=2012-12-31",
    "splits.cal_end=2013-12-31",
]


def _cfg(tmp_path=None, extra=()):
    over = [*SMALL, *extra]
    if tmp_path is not None:
        over.append(f"report.output_dir={tmp_path}")
    return load_config(ROOT / "configs/synthetic.yaml", over)


@pytest.fixture(scope="module")
def result():
    return run_backtest(_cfg())


# ---------------------------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------------------------


def test_splits_are_ordered_with_embargo():
    idx = pd.bdate_range("2015-01-01", "2018-12-31")
    sp = make_splits(idx, pd.Timestamp("2016-12-31").date(), pd.Timestamp("2017-12-31").date(),
                     None, embargo=5)  # fmt: skip
    pos = {d: i for i, d in enumerate(idx)}
    # The last training target (days t+1…t+5) must end before calibration starts.
    assert pos[sp.train[-1]] + 5 < pos[sp.cal[0]]
    assert pos[sp.cal[-1]] + 5 < pos[sp.test[0]]
    assert sp.test[0] > pd.Timestamp("2017-12-31")


def test_splits_too_small_raise():
    idx = pd.bdate_range("2015-01-01", periods=120)
    with pytest.raises(ValueError, match="too small"):
        make_splits(idx, idx[50].date(), idx[80].date(), None, 5)


# ---------------------------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------------------------


def test_backtest_outputs(result):
    info = result.info
    vol = result.vol
    for m in info["vol_methods"]:
        assert {f"{m}_lo", f"{m}_hi"} <= set(vol.columns)
        cov = ((vol["y"] >= vol[f"{m}_lo"]) & (vol["y"] <= vol[f"{m}_hi"])).mean()
        assert 0.8 < cov <= 1.0, m
    assert set(result.var) == {0.05, 0.01}
    for df in result.var.values():
        assert {"split_var", "rolling_var", "aci_var", "gaussian_var", "hist_var",
                "garch_var"} <= set(df.columns)  # fmt: skip
    assert "aci_vol" in result.alpha_paths
    assert {"split_es", "aci_es", "gaussian_es", "hist_es", "garch_es"} <= set(result.es.columns)
    es_cols = [c for c in result.es.columns if c.endswith("_es")]
    var_cols = [c.replace("_es", "_var") for c in es_cols]
    assert (result.es[es_cols].to_numpy() <= result.es[var_cols].to_numpy() + 1e-12).all()
    assert {"split", "rolling", "aci", "gaussian", "hist", "garch"} <= set(result.var_long.columns)
    assert np.isfinite(result.var_long[["split", "rolling", "aci"]].to_numpy()).all()
    assert result.contrib.sum(axis=1).to_numpy() == pytest.approx(1.0)
    names = result.info["strategy"]["names"]
    assert {"buy_hold", "vt_ewma", "vt_point", "vt_conformal", "alloc_erc"} <= set(names)
    for n in names:
        assert {f"{n}_exp", f"{n}_net"} <= set(result.strategies.columns)
    assert (result.strategies.filter(like="vt_").filter(like="_exp") <= 2.0 + 1e-12).all().all()
    assert info["latency"]["score_one_day"]["calls"] > 0
    assert vol.index[0] > pd.Timestamp("2013-12-31")


def test_backtest_has_no_lookahead():
    """Perturbing every price after test day t must not change any output dated ≤ t."""
    cfg = _cfg()
    world = generate_world(cfg.data.tickers, n_days=1500, seed=cfg.seed)

    def run(prices):
        market = build_market(prices, world.features, "equal", None, "synthetic")
        return run_backtest(cfg, Dataset(market, world.news))

    a = run(world.prices)
    t = a.vol.index[200]
    shocked = world.prices.copy()
    after = shocked.index > t  # price rows after t ↔ returns from t+1 on
    shocked.loc[after] *= np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.05, after.sum())))[
        :, None
    ]
    b = run(shocked)
    preds = [c for c in a.vol.columns if c != "y"]  # y is the forward-looking target itself
    pd.testing.assert_frame_equal(a.vol.loc[:t, preds], b.vol.loc[:t, preds])
    for alpha in a.var:
        cols = [c for c in a.var[alpha].columns if c != "r_next"]
        pd.testing.assert_frame_equal(a.var[alpha].loc[:t, cols], b.var[alpha].loc[:t, cols])
    # Regulatory extensions: ES, long-horizon VaR, Basel counts and capital, risk shares.
    for name, outcome in (("es", "r_next"), ("var_long", "r_fwd_long"), ("basel", None),
                          ("contrib", None)):  # fmt: skip
        fa, fb = getattr(a, name), getattr(b, name)
        cols = [c for c in fa.columns if c != outcome]
        pd.testing.assert_frame_equal(fa.loc[:t, cols], fb.loc[:t, cols])
    # Strategy exposures are decided at each close; their returns are realized the next day.
    exp_cols = [c for c in a.strategies.columns if c.endswith("_exp")]
    pd.testing.assert_frame_equal(a.strategies.loc[:t, exp_cols], b.strategies.loc[:t, exp_cols])
    # Sanity: the perturbation does change later outputs.
    assert not a.vol.loc[a.vol.index > t].equals(b.vol.loc[b.vol.index > t])


def test_save_load_roundtrip(result, tmp_path):
    result.save(tmp_path)
    back = BacktestResult.load(tmp_path)
    pd.testing.assert_frame_equal(back.vol, result.vol, check_freq=False)
    for name in ("es", "var_long", "basel", "contrib", "strategies"):
        pd.testing.assert_frame_equal(getattr(back, name), getattr(result, name), check_freq=False)
    for a in result.var:
        pd.testing.assert_frame_equal(back.var[a], result.var[a], check_freq=False)
    assert back.info["alpha"] == result.info["alpha"]


def test_report_files_and_tables(result, tmp_path):
    result.save(tmp_path)
    path = write_report(result, tmp_path)
    md = path.read_text()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert set(metrics["vol"]) == set(result.info["vol_methods"])
    assert {"es", "var_long", "basel", "contrib", "strategies"} <= set(metrics)
    assert "## Putting it to work" in md
    for heading in ("## Expected Shortfall", "-day 99% VaR", "## Basel traffic light",
                    "## Risk decomposition"):  # fmt: skip
        assert heading in md
    for f in ("vol_intervals", "rolling_coverage", "aci_alpha", "coverage_by_tercile", "var",
              "basel", "risk_contributions"):  # fmt: skip
        assert (tmp_path / "figures" / f"{f}.png").exists()
    # Every markdown table row has the same number of cells as its header.
    for block in re.findall(r"((?:^\|.*\|\n)+)", md, flags=re.M):
        rows = block.strip().splitlines()
        counts = {r.count("|") for r in rows}
        assert len(counts) == 1, rows[0]
    # Reported coverage matches the saved series.
    v = result.vol
    cov = ((v["y"] >= v["aci_lo"]) & (v["y"] <= v["aci_hi"])).mean()
    assert metrics["vol"]["aci"]["coverage"] == pytest.approx(cov)


def test_cli_backtest_and_report(tmp_path):
    over = []
    for s in [*SMALL, f"report.output_dir={tmp_path}", "report.figures=false"]:
        over += ["--set", s]
    assert main(["backtest", "-c", str(ROOT / "configs/synthetic.yaml"), *over]) == 0
    out = tmp_path / "synthetic"
    assert (out / "backtest_report.md").exists() and (out / "metrics.json").exists()
    assert main(["report", "-c", str(ROOT / "configs/synthetic.yaml"), *over]) == 0
    missing = ["--set", f"report.output_dir={tmp_path / 'nope'}"]
    assert main(["report", "-c", str(ROOT / "configs/synthetic.yaml"), *missing]) == 1
