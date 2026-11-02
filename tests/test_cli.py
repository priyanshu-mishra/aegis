import subprocess
import sys
from pathlib import Path

import yaml

from aegis.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_config_command_prints_resolved_yaml(capsys):
    rc = main(
        ["config", "-c", str(ROOT / "configs/synthetic.yaml"), "--set", "conformal.alpha=0.1"]
    )
    assert rc == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out["name"] == "synthetic"
    assert out["data"]["source"] == "synthetic"
    assert out["conformal"]["alpha"] == 0.1


def test_invalid_config_returns_1(capsys):
    assert main(["config", "--set", "conformal.alpha=2"]) == 1
    assert "Invalid configuration" in capsys.readouterr().err


def test_missing_config_file_returns_1():
    assert main(["config", "-c", "does/not/exist.yaml"]) == 1


def test_main_py_entry_point_runs():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "config"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "conformal:" in proc.stdout


def test_fetch_synthetic_runs_offline(capsys):
    assert main(["fetch", "-c", str(ROOT / "configs/synthetic.yaml")]) == 0
    assert "trading days" in capsys.readouterr().err
