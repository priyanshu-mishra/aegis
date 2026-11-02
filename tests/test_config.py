from pathlib import Path

import pytest
from pydantic import ValidationError

from aegis.config import AegisConfig, load_config, parse_override

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_valid():
    cfg = load_config()
    assert cfg.conformal.alpha == 0.05
    assert cfg.data.tickers == ["SPY", "TLT", "GLD", "USO"]
    assert cfg.splits.embargo_days >= cfg.risk.horizon_days


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    assert isinstance(load_config(path), AegisConfig)


def test_nested_defaults_are_not_shared():
    a, b = AegisConfig(), AegisConfig()
    a.data.tickers.append("QQQ")
    assert "QQQ" not in b.data.tickers


def test_overrides_are_yaml_typed():
    cfg = load_config(
        CONFIG_DIR / "default.yaml",
        ["conformal.alpha=0.1", "parser.mask_entities=false", "data.tickers=[SPY, TLT]"],
    )
    assert cfg.conformal.alpha == 0.1
    assert cfg.parser.mask_entities is False
    assert cfg.data.tickers == ["SPY", "TLT"]


def test_override_requires_equals():
    with pytest.raises(ValueError):
        parse_override("conformal.alpha")


@pytest.mark.parametrize(
    "overrides",
    [
        ["conformal.alpha=1.5"],  # alpha outside (0, 1)
        ["conformal.methods=[magic]"],  # unknown method
        ["data.tikers=[SPY]"],  # typo'd key is rejected, not ignored
        ["splits.cal_end=2015-01-01"],  # cal_end before train_end
        ["splits.embargo_days=2"],  # embargo shorter than horizon → leakage
        ["portfolio.weighting=custom"],  # custom without weights
        ["data.tickers=[SPY, SPY]"],  # duplicate tickers
    ],
)
def test_invalid_configs_rejected(overrides):
    with pytest.raises((ValidationError, ValueError)):
        load_config(None, overrides)


def test_custom_weights_must_match_tickers_and_sum_to_one():
    ok = {"SPY": 0.4, "TLT": 0.3, "GLD": 0.2, "USO": 0.1}
    cfg = AegisConfig.model_validate({"portfolio": {"weighting": "custom", "weights": ok}})
    assert cfg.portfolio.weights == ok

    with pytest.raises(ValidationError):
        AegisConfig.model_validate(
            {"portfolio": {"weighting": "custom", "weights": {"SPY": 0.5, "TLT": 0.5}}}
        )
    with pytest.raises(ValidationError):
        AegisConfig.model_validate(
            {"portfolio": {"weighting": "custom", "weights": {**ok, "SPY": 0.9}}}
        )
