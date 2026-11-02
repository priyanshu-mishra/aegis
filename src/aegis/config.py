"""Typed run configuration, loaded from YAML with optional dotted-key overrides.

Every module receives its settings from an ``AegisConfig`` instance; nothing reads YAML directly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ConformalMethod = Literal[
    "split_absolute", "split_normalized", "cqr", "rolling", "aci", "one_sided_var"
]


class _Strict(BaseModel):
    # Reject unknown keys so typos in YAML fail loudly instead of being ignored.
    model_config = ConfigDict(extra="forbid")


class SyntheticConfig(_Strict):
    n_days: int = Field(4000, gt=300)
    snr: float = Field(0.5, ge=0, description="Signal-to-noise ratio of news vs latent shock")
    news_per_day: float = Field(1.0, gt=0)


class DataConfig(_Strict):
    source: Literal["yfinance", "synthetic"] = "yfinance"
    tickers: list[str] = ["SPY", "TLT", "GLD", "USO"]
    feature_tickers: list[str] = ["^VIX"]
    start: dt.date = dt.date(2010, 1, 1)
    end: dt.date | None = None
    cache_dir: Path = Path("data")
    news_source: Literal["none", "fomc", "synthetic"] = "fomc"
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    @field_validator("tickers")
    @classmethod
    def _nonempty_unique(cls, v: list[str]) -> list[str]:
        if not v or len(set(v)) != len(v):
            raise ValueError("tickers must be non-empty and unique")
        return v


class PortfolioConfig(_Strict):
    weighting: Literal["equal", "inverse_vol", "custom"] = "equal"
    weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_weights(self) -> PortfolioConfig:
        if self.weighting == "custom":
            if not self.weights:
                raise ValueError("weighting='custom' requires weights")
            if abs(sum(self.weights.values()) - 1.0) > 1e-6:
                raise ValueError("custom weights must sum to 1")
        return self


class SplitConfig(_Strict):
    train_end: dt.date = dt.date(2017, 12, 31)
    cal_end: dt.date = dt.date(2019, 12, 31)
    test_end: dt.date | None = None
    embargo_days: int = Field(5, ge=0, description="Gap between splits; >= forecast horizon")
    refit: Literal["never"] = "never"  # periodic refits (e.g. yearly) are not implemented yet

    @model_validator(mode="after")
    def _ordered(self) -> SplitConfig:
        if not self.train_end < self.cal_end:
            raise ValueError("train_end must be before cal_end")
        if self.test_end is not None and not self.cal_end < self.test_end:
            raise ValueError("cal_end must be before test_end")
        return self


class ParserConfig(_Strict):
    backend: Literal["lexicon", "ollama"] = "lexicon"
    use_signals: bool = True  # False → price-only features (the ablation baseline)
    model: str = "qwen2.5:3b"
    ollama_url: str = "http://localhost:11434"
    timeout_s: float = Field(60.0, gt=0)
    prompt_version: str = "v2"  # v2: direction-named schema fields; bump to invalidate cache
    cache_path: Path = Path("data/slm_cache.sqlite")
    mask_entities: bool = True
    decay_half_life_days: float = Field(3.0, gt=0)
    clip_to_train: bool = True  # clip text features to their training-period range


class RiskConfig(_Strict):
    ewma_lambda: float = Field(0.94, gt=0, lt=1)
    horizon_days: int = Field(5, ge=1)
    point_model: Literal["har_ols", "ridge", "lightgbm"] = (
        "ridge"  # best out-of-sample on real and synthetic data (M3)
    )
    use_semantic_overlay: bool = True


class ConformalConfig(_Strict):
    alpha: float = Field(0.05, gt=0, lt=1)
    methods: list[ConformalMethod] = [
        "split_absolute",
        "split_normalized",
        "cqr",
        "rolling",
        "aci",
        "one_sided_var",
    ]
    rolling_window: int = Field(500, gt=20)
    aci_gamma: float = Field(0.005, gt=0)
    aci_floor_ratio: float | None = Field(
        0.2, gt=0, le=1, description="ACI quantile level floor as a fraction of α (None: off)"
    )
    online_base: Literal["absolute", "normalized", "cqr"] = Field(
        "absolute",
        description="Non-conformity score used by the online methods (rolling, aci). Absolute wins "
        "on real data: log-vol residuals are already near-homoskedastic, so scaling adds noise.",
    )
    var_alphas: list[float] = Field(
        [0.05, 0.01], description="Tail levels for the VaR backtest (0.05 → 95% VaR)"
    )

    es_alpha: float = Field(0.025, gt=0, lt=0.5, description="Expected Shortfall tail (FRTB: 2.5%)")
    long_horizon: int = Field(10, ge=2, description="Regulatory VaR horizon in days (Basel: 10)")
    long_alpha: float = Field(0.01, gt=0, lt=0.5, description="Tail for the long-horizon VaR")
    basel_window: int = Field(250, ge=50, description="Days per Basel traffic-light window")

    @field_validator("var_alphas")
    @classmethod
    def _var_alphas_in_range(cls, v: list[float]) -> list[float]:
        if not v or not all(0 < a < 0.5 for a in v):
            raise ValueError("var_alphas must be non-empty and each in (0, 0.5)")
        return v


class StrategyConfig(_Strict):
    """Portfolio rules that turn risk forecasts into position sizes (V2)."""

    target_vol: float = Field(0.10, gt=0, description="Annualized volatility target")
    max_leverage: float = Field(2.0, gt=0)
    cost_bps: float = Field(5.0, ge=0, description="Cost per unit of turnover, basis points")
    var_limit: float = Field(0.02, gt=0, description="Daily 99% VaR limit, fraction of capital")
    overshoot_window: int = Field(21, ge=5)
    overshoot_mult: float = Field(1.25, gt=1)


class ReportConfig(_Strict):
    output_dir: Path = Path("reports")
    figures: bool = True


class AegisConfig(_Strict):
    name: str = "default"
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> AegisConfig:
        if self.splits.embargo_days < self.risk.horizon_days:
            raise ValueError(
                f"splits.embargo_days ({self.splits.embargo_days}) must be >= "
                f"risk.horizon_days ({self.risk.horizon_days}) to avoid target leakage"
            )
        src, news = self.data.source, self.data.news_source
        if (src, news) in {("yfinance", "synthetic"), ("synthetic", "fomc")}:
            raise ValueError(
                f"data.news_source={news!r} cannot be paired with data.source={src!r}: "
                "synthetic news only lines up with synthetic prices, and FOMC text with real ones"
            )
        w = self.portfolio.weights
        if w and set(w) != set(self.data.tickers):
            raise ValueError("portfolio.weights keys must match data.tickers")
        return self


def _set_dotted(tree: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    node = tree
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override '{dotted_key}': '{key}' is not a section")
    node[leaf] = value


def parse_override(item: str) -> tuple[str, Any]:
    """Parse ``key.path=value``; the value is YAML-decoded so ``0.1``, ``true``, ``[a, b]`` work."""
    if "=" not in item:
        raise ValueError(f"override must look like key.path=value, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> AegisConfig:
    """Load YAML config (defaults if ``path`` is None), then apply ``key.path=value`` overrides."""
    tree: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            tree = yaml.safe_load(f) or {}
    for item in overrides or []:
        _set_dotted(tree, *parse_override(item))
    return AegisConfig.model_validate(tree)
