import importlib.util
from pathlib import Path

import pytest

from aegis.bench import run_bench
from aegis.cli import DEMO_OVERRIDES, main
from aegis.config import load_config
from aegis.slm_parser import OllamaParser

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_offline(tmp_path):
    over = ["--set", f"report.output_dir={tmp_path}", "--set", "report.figures=false",
            "--set", f"parser.cache_path={tmp_path / 'c.sqlite'}"]  # fmt: skip
    assert main(["demo", *over]) == 0
    md = (tmp_path / "demo" / "backtest_report.md").read_text()
    assert "Aegis backtest: demo" in md


def test_bench_without_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr(OllamaParser, "ping", lambda self: False)
    cfg = load_config(ROOT / "configs/synthetic.yaml",
                      [*DEMO_OVERRIDES, f"parser.cache_path={tmp_path / 'c.sqlite'}"])  # fmt: skip
    r = run_bench(cfg)
    assert r["slm_parse"] is None
    assert r["test_days"] > 1000 and r["score_one_day"]["p50_ms"] > 0
    assert r["lexicon_parse"]["n"] > 20


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="streamlit not installed")
def test_dashboard_renders_every_view(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    over = ["--set", f"report.output_dir={tmp_path}", "--set", "report.figures=false",
            "--set", f"parser.cache_path={tmp_path / 'c.sqlite'}"]  # fmt: skip
    assert main(["demo", *over]) == 0
    monkeypatch.setenv("AEGIS_REPORTS", str(tmp_path))
    at = AppTest.from_file(str(ROOT / "dashboard/app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Aegis · demo"
    for view in ("Coverage", "VaR", "Text ablation", "Latency"):
        at.segmented_control[0].set_value(view).run()
        assert not at.exception, (view, at.exception)
