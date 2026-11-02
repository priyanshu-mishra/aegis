import datetime as dt
import json

import httpx
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from aegis.config import ParserConfig
from aegis.data.news import NewsItem, NewsStore, market_close_utc
from aegis.features import aggregate_signals
from aegis.slm_parser import (
    CachedParser,
    LexiconParser,
    OllamaParser,
    RiskSignal,
    build_parser,
    mask_entities,
    parse_items,
)

TICKERS = ["SPY", "TLT", "GLD", "USO"]


def _valid_signal(**over) -> dict:
    d = {
        "sentiment_score": 0.2,
        "macro_shock_vector": {"rates": 0.5, "inflation": 0, "growth": 0, "energy": 0,
                               "geopolitics": 0, "credit": 0},
        "volatility_premium_signal": {"direction": "up", "magnitude": 0.3},
        "asset_impacts": {"SPY": -0.2, "TLT": -0.5},
        "confidence": 0.8,
        "evidence": ["rates up"],
    }  # fmt: skip
    d.update(over)
    return d


# ---------------------------------------------------------------------------------------------
# Schema and masking
# ---------------------------------------------------------------------------------------------


def test_schema_validates_ranges_and_rejects_extras():
    RiskSignal.model_validate(_valid_signal())
    for bad in (
        _valid_signal(sentiment_score=1.5),
        _valid_signal(confidence=-0.1),
        _valid_signal(asset_impacts={"SPY": 2.0}),
        _valid_signal(evidence=["a", "b", "c", "d"]),
        {**_valid_signal(), "surprise": 1},
    ):
        with pytest.raises(ValidationError):
            RiskSignal.model_validate(bad)


def test_flat_fields():
    flat = RiskSignal.model_validate(_valid_signal()).flat()
    assert flat["shock_rates"] == 0.5 and flat["vol_premium"] == 0.3
    assert flat["impact_TLT"] == -0.5


def test_mask_entities():
    text = "On March 15, 2020 Powell said COVID-19 hit activity in 2020; see 3/15/20 and May 2021."
    masked = mask_entities(text)
    assert masked == (
        "On [DATE] [PERSON] said [EVENT] hit activity in [YEAR]; see [DATE] and [DATE]."
    )
    assert mask_entities("Inflation rose sharply.") == "Inflation rose sharply."


# ---------------------------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, factor, sign",
    [
        ("The Committee decided to raise the target range for the federal funds rate.", "rates", 1),
        ("The Committee decided to lower the target range.", "rates", -1),
        ("Credit spreads tighten as funding conditions stabilize.", "credit", -1),
        ("Credit spreads widen sharply amid default worries.", "credit", 1),
        ("Oil prices jump on supply disruption.", "energy", 1),
        ("Inflation has eased somewhat but remains elevated.", "inflation", 0),
        ("Economic activity has continued to strengthen.", "growth", 1),
        ("Tensions ease after ceasefire agreement.", "geopolitics", -1),
    ],
)
def test_lexicon_directions(text, factor, sign):
    v = LexiconParser(TICKERS).parse(text).macro_shock_vector.model_dump()[factor]
    assert np.sign(round(v, 6)) == sign


def test_lexicon_negation_and_intensity():
    p = LexiconParser(TICKERS)
    assert p.parse("The Fed will not raise the target range.").macro_shock_vector.rates < 0
    strong = p.parse("Oil prices jump sharply.").macro_shock_vector.energy
    mild = p.parse("Oil prices jump slightly.").macro_shock_vector.energy
    assert strong > mild > 0


def test_lexicon_cross_factor_words_do_not_collide():
    # "tighter policy" is about rates; "spreads tighten" is about credit, with opposite risk.
    s = LexiconParser(TICKERS).parse("Tighter policy expected. Credit spreads tighten.")
    assert s.macro_shock_vector.rates > 0 and s.macro_shock_vector.credit < 0


def test_lexicon_empty_and_deterministic():
    p = LexiconParser(TICKERS)
    e = p.parse("")
    assert e.confidence == 0.05 and not any(e.macro_shock_vector.model_dump().values())
    text = "Crude slumps as supply glut deepens; recession worries grow."
    assert p.parse(text) == p.parse(text)
    impacts = p.parse(text).asset_impacts
    assert set(impacts) == set(TICKERS) and impacts["USO"] < 0


# ---------------------------------------------------------------------------------------------
# Ollama backend (mocked HTTP)
# ---------------------------------------------------------------------------------------------


def _ollama(responses: list, tickers=TICKERS):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]})
        calls.append(json.loads(request.content))
        r = responses.pop(0)
        if isinstance(r, int):
            return httpx.Response(r)
        return httpx.Response(200, json={"message": {"content": r}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://ollama")
    return OllamaParser(tickers=tickers, client=client), calls


def test_ollama_valid_response_and_request_shape():
    p, calls = _ollama([json.dumps(_valid_signal())])
    sig = p.parse("Yields climb.")
    assert sig.macro_shock_vector.rates == 0.5
    assert set(sig.asset_impacts) == set(TICKERS)  # normalised to the ticker set
    assert sig.asset_impacts["GLD"] == 0.0
    req = calls[0]
    assert req["options"]["temperature"] == 0 and req["stream"] is False
    assert req["format"]["title"] == "RiskSignal"
    assert req["messages"][-1]["content"] == "Yields climb."
    assert p.fallbacks == 0 and p.ping()


def test_ollama_retries_once_on_invalid_then_succeeds():
    bad = json.dumps(_valid_signal(sentiment_score=3))
    p, calls = _ollama([bad, json.dumps(_valid_signal())])
    assert p.parse("x").sentiment_score == 0.2
    assert p.retries == 1 and p.fallbacks == 0
    assert "invalid" in calls[1]["messages"][-1]["content"]


def test_ollama_falls_back_to_lexicon():
    p, _ = _ollama(["not json", "{}"])
    text = "Oil prices jump sharply."
    assert p.parse(text) == LexiconParser(TICKERS).parse(text)
    assert p.fallbacks == 1
    p2, _ = _ollama([500])
    assert p2.parse(text).macro_shock_vector.energy > 0 and p2.fallbacks == 1


def test_build_parser_unreachable_ollama_raises(tmp_path):
    cfg = ParserConfig(backend="ollama", ollama_url="http://127.0.0.1:9", timeout_s=0.5,
                       cache_path=tmp_path / "c.sqlite")  # fmt: skip
    with pytest.raises(RuntimeError, match="not reachable"):
        build_parser(cfg, TICKERS)


# ---------------------------------------------------------------------------------------------
# Cache and batch parsing
# ---------------------------------------------------------------------------------------------


class _Counting(LexiconParser):
    calls: int = 0

    def parse(self, text):
        self.calls += 1
        return super().parse(text)


def test_cache_hits_and_key_includes_prompt_version(tmp_path):
    inner = _Counting(TICKERS)
    c = CachedParser(inner, tmp_path / "c.sqlite")
    a = c.parse("Oil prices jump.")
    b = c.parse("Oil prices jump.")
    assert a == b and inner.calls == 1 and c.hits == 1 and c.misses == 1
    c.close()
    c2 = CachedParser(inner, tmp_path / "c.sqlite")  # persists across instances
    c2.parse("Oil prices jump.")
    assert inner.calls == 1
    c3 = CachedParser(inner, tmp_path / "c.sqlite", prompt_version="v2")
    c3.parse("Oil prices jump.")
    assert inner.calls == 2


def _item(i, when, text):
    return NewsItem(f"n{i}", when, "t", text, "")


def test_parse_items_masks_by_default():
    seen = []

    class Spy(LexiconParser):
        def parse(self, text):
            seen.append(text)
            return super().parse(text)

    t = market_close_utc(dt.date(2024, 3, 5)) - dt.timedelta(hours=2)
    df = parse_items(Spy(TICKERS), [_item(0, t, "Powell spoke in 2024.")])
    assert seen == ["[PERSON] spoke in [YEAR]."]
    assert {"id", "published_at", "shock_rates", "confidence"} <= set(df.columns)


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------


def _parsed(rows):
    base = {f"shock_{f}": 0.0 for f in
            ("rates", "inflation", "growth", "energy", "geopolitics", "credit")}  # fmt: skip
    return pd.DataFrame([{**base, "sentiment": 0.0, "vol_premium": 0.0, **r} for r in rows])


def test_aggregation_level_math_and_timing():
    days = pd.bdate_range("2024-03-04", periods=6)
    before = market_close_utc(days[1]) - dt.timedelta(hours=1)
    after = market_close_utc(days[1]) + dt.timedelta(hours=1)  # rolls to days[2]
    store = NewsStore([_item(0, before, "a"), _item(1, after, "b")])
    parsed = _parsed([
        {"id": "n0", "confidence": 1.0, "shock_energy": 0.8},
        {"id": "n1", "confidence": 0.5, "shock_rates": -0.6},
    ])  # fmt: skip
    agg = aggregate_signals(parsed, store, days, half_life_days=1.0, prior_weight=0.5)
    # Energy: one observation of 0.8 with weight 1 on day 1 → 0.8 / (1 + 0.5); then decays by ½.
    assert agg["shock_energy"].iloc[0] == 0
    assert agg["shock_energy"].iloc[1] == pytest.approx(0.8 / 1.5)
    assert agg["shock_energy"].iloc[2] == pytest.approx(0.4 / (0.5 + 0.5))
    # The rates item was published after day 1's close, so it first counts on day 2.
    assert agg["shock_rates"].iloc[1] == 0
    assert agg["shock_rates"].iloc[2] == pytest.approx(0.5 * -0.6 / (0.5 + 0.5))
    assert agg["abs_shock_rates"].iloc[2] == pytest.approx(0.3)
    # An unmentioned factor in the rates item doesn't drag the energy level toward zero.
    assert agg["shock_energy"].iloc[2] > 0.3
    assert agg["intensity"].iloc[1] == 1 and agg["intensity"].iloc[2] == pytest.approx(1.5)


def test_aggregation_ignores_future_items():
    days = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(0)
    items, rows = [], []
    for i in range(40):
        t = market_close_utc(days[rng.integers(30)]) - dt.timedelta(hours=2)
        items.append(_item(i, t, "x"))
        rows.append({"id": f"n{i}", "confidence": 0.7, "shock_growth": rng.uniform(-1, 1)})
    cut = days[15]
    keep = [it for it in items if it.published_at < market_close_utc(cut)]
    a = aggregate_signals(_parsed(rows), NewsStore(items), days)
    b = aggregate_signals(_parsed(rows), NewsStore(keep), days)
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])


@pytest.mark.llm
def test_live_ollama_returns_valid_signal(tmp_path):
    cfg = ParserConfig(backend="ollama", cache_path=tmp_path / "c.sqlite")
    parser = build_parser(cfg, TICKERS)  # raises if the server or model is missing
    sig = parser.parse("Oil prices jump sharply on supply disruption and output cuts.")
    assert set(sig.asset_impacts) == set(TICKERS)
    assert sig.macro_shock_vector.energy > 0
    assert parser.inner.fallbacks == 0
    parser.close()


# ---------------------------------------------------------------------------------------------
# Direction-named schema (v2) and clipping
# ---------------------------------------------------------------------------------------------


def test_model_facing_schema_uses_direction_names():
    from aegis.slm_parser import FEW_SHOT, ShockVector

    props = RiskSignal.model_json_schema()["$defs"]["ShockVector"]["properties"]
    expected = ["rate_pressure", "inflation_pressure", "growth_momentum",
                "energy_price_pressure", "geopolitical_tension", "credit_stress"]  # fmt: skip
    assert list(props) == expected
    assert all("+1" in p["description"] and "-1" in p["description"] for p in props.values())
    # Model output arrives under aliases; internal code and the cache use field names.
    sv = ShockVector.model_validate({"credit_stress": -0.5})
    assert sv.credit == -0.5 and sv.model_dump()["credit"] == -0.5
    sig = RiskSignal.model_validate(FEW_SHOT[2][1])
    assert RiskSignal.model_validate_json(sig.model_dump_json()) == sig  # cache roundtrip


def test_few_shots_are_valid_and_balanced():
    from aegis.slm_parser import FEW_SHOT

    signs = []
    for _, answer in FEW_SHOT:
        sv = RiskSignal.model_validate(answer).macro_shock_vector.model_dump()
        signs += [np.sign(v) for v in sv.values() if v]
    assert signs.count(1) >= 3 and signs.count(-1) >= 3  # no one-directional bias


def test_default_prompt_version_separates_cache():
    assert ParserConfig().prompt_version == "v2"


def test_clip_to_training_range():
    from aegis.features import clip_to_training_range

    idx = pd.bdate_range("2020-01-01", periods=10)
    sig = pd.DataFrame({"a": np.arange(10.0), "b": -np.arange(10.0)}, index=idx)
    out = clip_to_training_range(sig, idx[:4])  # train range: a ∈ [0, 3], b ∈ [−3, 0]
    assert out["a"].max() == 3 and out["b"].min() == -3
    pd.testing.assert_frame_equal(out.iloc[:4], sig.iloc[:4])
