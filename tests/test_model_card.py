import re
from pathlib import Path

import numpy as np

from aegis.cli import main
from aegis.config import load_config
from aegis.model_card import current_monitoring, render

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site/public/data"


def test_committed_model_card_matches_committed_data():
    # The report is generated; a stale copy would quote numbers the site no longer shows.
    cfg = load_config(ROOT / "configs/default.yaml")
    assert (ROOT / "docs/MODEL_CARD.md").read_text() == render(DATA, cfg), (
        "docs/MODEL_CARD.md is stale: run `python main.py model-card -c configs/default.yaml`"
    )


def test_model_card_has_every_section_and_no_missing_values():
    text = render(DATA, load_config(ROOT / "configs/default.yaml"))
    for n in range(1, 10):
        assert f"\n## {n}. " in text
    assert not re.search(r"\bnan\b|\bNone\b", text)


def test_monitoring_statistics_are_in_range():
    mon = current_monitoring(DATA)
    assert 0 <= mon["coverage_63d"] <= 1
    assert 0 <= mon["kupiec_p_250d"] <= 1
    assert mon["breaches_250d"] >= 0
    assert 0 <= mon["coverage_fired"] <= 1 and 0 <= mon["basel_fired"] <= 1
    assert np.isfinite(mon["coverage_63d"])


def test_cli_writes_model_card(tmp_path):
    out = tmp_path / "card.md"
    assert main(["model-card", "-c", str(ROOT / "configs/default.yaml"), "--out", str(out)]) == 0
    assert out.read_text().startswith("# Model validation report")
