import itertools
from pathlib import Path

import pytest

from aegis.ablation import SNR_SWEEP, run_ablation, write_ablation
from aegis.cli import main
from aegis.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("abl")
    return load_config(ROOT / "configs/synthetic.yaml", [
        "data.synthetic.n_days=1500", "splits.train_end=2012-12-31", "splits.cal_end=2013-12-31",
        f"parser.cache_path={tmp / 'cache.sqlite'}", f"report.output_dir={tmp}",
    ])  # fmt: skip


@pytest.fixture(scope="module")
def table(cfg):
    return run_ablation(cfg)


def test_ablation_variants(table):
    names = list(table.index)
    assert names[0] == "prices only" and names[-1].startswith("+ oracle")
    assert len(names) == 2 + len(SNR_SWEEP)
    assert (table["aci_coverage"] > 0.85).all()


def test_shock_recovery_rises_with_snr(table):
    rec = table["shock_recovery"].iloc[1:-1].to_numpy()  # SNR sweep rows, increasing SNR
    assert abs(rec[0]) < 0.15  # SNR 0: headlines carry no information
    assert rec[-1] > 0.8  # near noise-free headlines are read almost perfectly
    assert all(b >= a - 0.05 for a, b in itertools.pairwise(rec))


def test_write_ablation(table, cfg, tmp_path):
    path = write_ablation(table, cfg, tmp_path)
    md = path.read_text()
    assert "prices only" in md and "oracle" in md and (tmp_path / "ablation.json").exists()


def test_cli_parse_synthetic(cfg):
    config = str(ROOT / "configs/synthetic.yaml")
    cache = f"parser.cache_path={cfg.parser.cache_path}"
    args = ["parse", "-c", config, "--set", "data.synthetic.n_days=400", "--set", cache]
    assert main(args) == 0
