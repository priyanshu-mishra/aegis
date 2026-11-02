import json
from pathlib import Path

import numpy as np
import pytest

from aegis.backtest import BacktestResult, run_backtest
from aegis.cli import main
from aegis.config import load_config
from aegis.conformal_ncm import AbsoluteResidual, Forecast, SplitConformal
from aegis.export import SCHEMA_VERSION, export_site, golden_values, lexicon_rules

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["data.synthetic.n_days=1500", "splits.train_end=2012-12-31", "splits.cal_end=2013-12-31"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("export")
    cfg = load_config(ROOT / "configs/synthetic.yaml", SMALL)
    res = run_backtest(cfg)
    run_dir = tmp / "reports" / "synthetic"
    res.save(run_dir)
    out = tmp / "data"
    export_site(run_dir, out)
    return res, out


def _load(out, name):
    return json.loads((out / f"{name}.json").read_text())


def test_expected_files_and_strict_json(exported):
    _, out = exported
    names = {p.stem for p in out.glob("*.json")}
    assert {"meta", "metrics", "vol", "calibration", "golden", "aci_alpha", "var", "es",
            "var_long", "basel", "contrib", "strategies", "lexicon"} <= names  # fmt: skip
    for p in out.glob("*.json"):
        text = p.read_text()
        assert "NaN" not in text and "Infinity" not in text, p.name
        json.loads(text)


def test_columnar_frames_are_aligned(exported):
    res, out = exported
    vol = _load(out, "vol")
    n = len(vol["dates"])
    assert n == len(res.vol)
    assert all(len(v) == n for v in vol["series"].values())
    assert vol["dates"][0] == res.vol.index[0].strftime("%Y-%m-%d")
    var = _load(out, "var")
    assert set(var) == {"0.05", "0.01"}
    assert all(len(v) == len(var["0.01"]["dates"]) for v in var["0.01"]["series"].values())


def test_meta(exported):
    _, out = exported
    meta = _load(out, "meta")
    assert meta["schema_version"] == SCHEMA_VERSION
    assert {"data", "splits", "alpha", "horizon", "vol_methods"} <= set(meta)


def test_golden_values_reproducible_from_exported_json(exported):
    """A re-implementation that reads only the JSON must get Python's split threshold."""
    res, out = exported
    cal = _load(out, "calibration")["series"]
    vol = _load(out, "vol")["series"]
    golden = _load(out, "golden")
    yc, mc = np.array(cal["y"], float), np.array(cal["mu"], float)
    for a, g in golden["levels"].items():
        q = SplitConformal(AbsoluteResidual(), float(a)).fit(Forecast(mu=mc), yc).q
        assert q == pytest.approx(g["split_q"], abs=1e-12)
        yt, mt = np.array(vol["y"], float), np.array(vol["mu"], float)
        assert np.mean(np.abs(yt - mt) <= q) == pytest.approx(g["split_coverage"])
        assert 0.8 < g["aci_coverage"] <= 1.0
    assert golden == json.loads(json.dumps(golden_values(res)))


def test_missing_optional_inputs_leave_existing_files(exported, tmp_path):
    res, _ = exported
    run_dir = tmp_path / "reports" / "synthetic"
    res.save(run_dir)
    out = tmp_path / "data"
    out.mkdir()
    (out / "ablation_ollama.json").write_text('{"kept": true}')
    export_site(run_dir, out)
    assert json.loads((out / "ablation_ollama.json").read_text()) == {"kept": True}


def test_lexicon_rules_compile_and_cover_factors():
    import re

    rules = lexicon_rules()
    assert set(rules["direction"]) == {"rates", "inflation", "growth", "energy", "geopolitics",
                                       "credit"}  # fmt: skip
    for patterns in rules["direction"].values():
        for p, sign in patterns:
            re.compile(p, re.I)
            assert sign in (1, -1)
    assert rules["aliases"]["credit"] == "credit_stress"


def test_cli_export_site(tmp_path):
    over = []
    for s in [*SMALL, f"report.output_dir={tmp_path / 'reports'}", "report.figures=false"]:
        over += ["--set", s]
    cfg = str(ROOT / "configs/synthetic.yaml")
    assert main(["backtest", "-c", cfg, *over]) == 0
    assert main(["export-site", "-c", cfg, *over, "--out", str(tmp_path / "data")]) == 0
    assert (tmp_path / "data" / "prices.json").exists()
    res = BacktestResult.load(tmp_path / "reports" / "synthetic")
    assert res.vol_cal is not None
