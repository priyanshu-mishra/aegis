"""Component 1: semantic parser turning financial text into a validated :class:`RiskSignal`.

Backends share the :class:`Parser` protocol:

* :class:`LexiconParser`: deterministic phrase rules. Default, offline, and the baseline any SLM
  must beat.
* :class:`OllamaParser`: a local SLM via Ollama with JSON-schema constrained decoding, validated
  by pydantic, retried once on a validation error, then falling back to the lexicon parser. The
  pipeline never fails because the model did.

:class:`CachedParser` wraps any backend with a SQLite cache keyed by
sha256(backend, model, prompt version, masked text), which doubles as an audit log.

Sign convention for ``macro_shock_vector`` (shared by all backends): positive means rising rates,
rising inflation, stronger growth, rising energy prices, escalating geopolitical tension, and
widening credit stress.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aegis.data.news import NewsItem

log = logging.getLogger(__name__)

FACTORS = ("rates", "inflation", "growth", "energy", "geopolitics", "credit")

# --------------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------------


class ShockVector(BaseModel):
    """Signed macro shocks. Python names are internal; the JSON the model sees uses aliases that
    name each factor by its positive direction, so the model never has to remember a sign
    convention (with neutral names, a 3B model read "credit" as credit *health* and inverted it).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    rates: float = Field(
        0.0,
        ge=-1,
        le=1,
        alias="rate_pressure",
        description=(
            "+1: rates rising or policy tightening (hikes, hawkish). "
            "-1: rates falling or policy easing (cuts, dovish, asset purchases)."
        ),
    )
    inflation: float = Field(
        0.0,
        ge=-1,
        le=1,
        alias="inflation_pressure",
        description=("+1: inflation rising or elevated. -1: inflation cooling or below target."),
    )
    growth: float = Field(
        0.0,
        ge=-1,
        le=1,
        alias="growth_momentum",
        description=("+1: economy strengthening. -1: economy weakening, contracting, layoffs."),
    )
    energy: float = Field(
        0.0,
        ge=-1,
        le=1,
        alias="energy_price_pressure",
        description=(
            "+1: oil/energy prices rising or supply tightening. -1: energy prices falling, glut."
        ),
    )
    geopolitics: float = Field(
        0.0,
        ge=-1,
        le=1,
        alias="geopolitical_tension",
        description=(
            "+1: tension escalating (war, sanctions, conflict). -1: tension easing (ceasefire, "
            "diplomatic progress)."
        ),
    )
    credit: float = Field(
        0.0,
        ge=-1,
        le=1,
        alias="credit_stress",
        description=(
            "+1: credit stress rising (spreads widen, defaults, bank stress). "
            "-1: credit stress falling (spreads tighten, markets calm, funding stabilizes)."
        ),
    )


class VolPremiumSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["up", "down", "neutral"]
    magnitude: float = Field(ge=0, le=1)


class RiskSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    sentiment_score: float = Field(ge=-1, le=1)
    macro_shock_vector: ShockVector
    volatility_premium_signal: VolPremiumSignal
    asset_impacts: dict[str, float]
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("asset_impacts")
    @classmethod
    def _impacts_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        bad = {k: x for k, x in v.items() if not -1 <= x <= 1}
        if bad:
            raise ValueError(f"asset impacts outside [-1, 1]: {bad}")
        return v

    def flat(self) -> dict[str, float]:
        """Numeric fields as a flat dict (used to build feature columns)."""
        vp = self.volatility_premium_signal
        sign = {"up": 1.0, "down": -1.0, "neutral": 0.0}[vp.direction]
        out = {"sentiment": self.sentiment_score, "vol_premium": sign * vp.magnitude,
               "confidence": self.confidence}  # fmt: skip
        out.update({f"shock_{k}": v for k, v in self.macro_shock_vector.model_dump().items()})
        out.update({f"impact_{k}": v for k, v in self.asset_impacts.items()})
        return out


# --------------------------------------------------------------------------------------------
# Masking (memorisation-leakage control)
# --------------------------------------------------------------------------------------------

_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
)
_MASKS = [
    (re.compile(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b"), "[DATE]"),
    (re.compile(rf"\b{_MONTHS}\s+(?:19|20)\d{{2}}\b"), "[DATE]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b"), "[DATE]"),
    (re.compile(r"\b(?:19|20)\d{2}\b"), "[YEAR]"),
    (re.compile(r"\b(?:COVID-19|COVID|coronavirus|SARS-CoV-2)\b", re.I), "[EVENT]"),
    (re.compile(r"\b(?:Bernanke|Yellen|Powell|Warsh|Hassett|Greenspan|Volcker)\b"), "[PERSON]"),
]


def mask_entities(text: str) -> str:
    """Replace dates, years, named events and Fed chairs with placeholders.

    An SLM pretrained on later text can otherwise recall what followed a dated event, which
    would leak the future into a backtest. Generic economic language is left intact.
    """
    for pattern, repl in _MASKS:
        text = pattern.sub(repl, text)
    return text


# --------------------------------------------------------------------------------------------
# Lexicon backend
# --------------------------------------------------------------------------------------------

# Factor-specific direction phrases: (+1) pushes the factor up, (−1) down.
_DIRECTION = {
    "rates": [
        (r"\bhike[sd]?\b|\bhiking\b", 1),
        (r"\b(?:rais|increas)(?:e|ed|es|ing) the target range\b", 1),
        (
            r"\btighter (?:policy|monetary)|\btightening\b(?! (?:of )?(?:credit|lending|spreads?))",
            1,
        ),
        (r"\bhawkish\b|\bfurther increases\b|\badditional tightening\b", 1),
        (r"\bongoing increases in the target range\b|\breduc\w* (?:its )?holdings\b", 1),
        (r"\byields? (?:climb|rise|rose|jump|surge|move higher)\w*", 1),
        (r"\b(?:lower|reduc|cut)(?:s|ed|ing|e)? the target range\b", -1),
        (r"\brate cuts?\b|\bcut (?:interest )?rates\b|\bdovish\b", -1),
        (r"\beas(?:e|ing) (?:policy|monetary)|policy could ease|\bstart to easing\b", -1),
        (r"\bexceptionally low levels\b|\bhighly accommodative\b|\basset purchases\b", -1),
        (r"\byields? (?:fall|fell|drop|decline|move lower|sink)\w*|\byields? \w+ lower\b", -1),
    ],
    "inflation": [
        (
            r"\binflation (?:fears|pressures? intensif\w*|accelerat\w*|rose|rises|rising|jump\w*|"
            r"remains? elevated|picked up|has risen)",
            1,
        ),
        (r"\bprices? (?:rise|rose|jump|climb|surge)\w*|\bcost pressures? intensif\w*", 1),
        (r"\bfaster than expected\b|\brunning hot\b|\binput costs rose\b", 1),
        # In Fed statements "remain(s) elevated" almost always refers to inflation.
        (
            r"\b(?:remains?|remained) elevated\b(?<!inflation remains elevated)"
            r"(?<!inflation remain elevated)",
            1,
        ),
        (r"\binflation (?:cool|eas|moderat|declin|slow|fell|has eased)\w*|\bdisinflation\b", -1),
        (r"\bprice (?:growth slows|pressures? ease)\w*|\bprices? (?:fall|fell|declin)\w*", -1),
        (
            r"\binflation has (?:declined|moved down)"
            r"|\brunning below (?:the Committee's )?2 percent",
            -1,
        ),
    ],
    "growth": [
        (r"\bexpand(?:s|ed|ing)?\b|\bexpansion\b|\bsolid (?:pace|rate)\b", 1),
        (r"\bstrong (?:hiring|growth|demand|job)|\brobust\b|\bbeat forecasts\b", 1),
        (r"\b(?:activity|output|manufacturing\s+\w+) accelerat\w*|\bnew orders surge\b", 1),
        (r"\bjob gains have been (?:strong|solid|robust)\b|\bimproving demand\b", 1),
        (r"\b(?:activity|labor market|economy) (?:has )?(?:continued to )?strengthen\w*", 1),
        (r"\bpicked up\b|\brising at a (?:solid|strong) rate\b", 1),
        (r"\bgrowth slow\w*|\bslowdown\b|\bcontract(?:s|ed|ion)\b|\brecession\b", -1),
        (r"\bweak(?:en|ened|ening|ness)?\b|\blayoffs?\b|\bhiring stalled\b|\bdisappointing\b", -1),
        (r"\bdeteriorat\w*(?![^.]{0,60}\babating\b)|\bsoftened\b|\bhas slowed\b", -1),
        (
            r"\bunemployment (?:rate )?(?:rose|rising|climbed|has risen|moved up)"
            r"|\bclaims climbed\b",
            -1,
        ),
    ],
    "energy": [
        (
            r"\b(?:oil|crude|energy prices?)\b[^.]{0,30}"
            r"\b(?:jump|rall|surge|soar|rise|rose|climb)\w*",
            1,
        ),
        (r"\bsupply disruption\b|\boutput cuts?\b|\bshortages?\b|\btightens supply\b", 1),
        (
            r"\b(?:oil|crude|energy prices?)\b[^.]{0,30}\b(?:slump|fall|fell|drop|plung|declin)\w*",
            -1,
        ),
        (r"\bsupply glut\b|\bboost output\b|\bample supply\b|\binventories rose\b", -1),
    ],
    "geopolitics": [
        (r"\bescalat\w*|\bconflict\b|\bwar\b|\bmilitary\b|\bsanctions?\b|\bretaliation\b", 1),
        (r"\btensions? (?:rise|rose|escalate|intensif)\w*|\btalks collapse\b", 1),
        (r"\bceasefire\b|\btensions? eas\w*|\bdiplomatic (?:progress|breakthrough)\b", -1),
        (r"\bde-?escalat\w*|\bresume negotiations\b|\breduces conflict risk\b", -1),
    ],
    "credit": [
        (r"\bspreads? widen\w*|\bdefault worries\b|\bdefaults?\b|\bdowngrades? mount\b", 1),
        (r"\bbank stress\b|\b(?:credit|funding) (?:stress|strains)\b|\bbond selloff\b", 1),
        (r"\btightened (?:lending )?standards\b|\bfunding costs rose\b|\bpulled money\b", 1),
        (
            r"\bspreads? (?:tighten|narrow)\w*|\b(?:credit markets?|funding conditions) "
            r"(?:calm|stabiliz)\w*",
            -1,
        ),
        (r"\bstress fades\b|\bwell capitalized\b|\bissuance resumed\b|\bdeposits stabilize\b", -1),
    ],
}
_DIRECTION_RE = {k: [(re.compile(p, re.I), s) for p, s in v] for k, v in _DIRECTION.items()}

_POSITIVE = re.compile(
    r"\b(?:gains?|strong|strengthen\w*|robust|improv\w*|expand\w*|stabiliz\w*|calm\w*|progress|"
    r"beat|resilient|solid|recover\w*|breakthrough|ceasefire|upside)\b", re.I)  # fmt: skip
_NEGATIVE = re.compile(
    r"\b(?:weak\w*|declin\w*|fears?|worr\w*|stress|defaults?|crisis|recession|slump\w*|"
    r"conflict|uncertain\w*|layoffs?|disappoint\w*|selloff|contract\w*|downgrades?|harm\w*|"
    r"disrupt\w*|collapse|shortages?|rattle)\b", re.I)  # fmt: skip
_RISK_UP = re.compile(
    r"\b(?:uncertain\w*|fears?|rattl\w*|volatil\w*|stress|crisis|selloff|turmoil|escalat\w*|"
    r"shocks?|worr\w*|sought safety|harm\w*|disrupt\w*)\b", re.I)  # fmt: skip
_RISK_DOWN = re.compile(
    r"\b(?:calm\w*|stabiliz\w*|ceasefire|resilient|tensions? eas\w*|fades|unchanged)\b", re.I
)
_STRONG = re.compile(r"\b(?:sharply|significantly|substantially|aggressive\w*|tremendous)\b", re.I)
_MODERATE = re.compile(r"\b(?:modestly|slightly|somewhat|moderately|gradual\w*)\b", re.I)
_NEGATION = re.compile(r"\b(?:not|no|never|without|n't)\b(?:\W+\w+){0,3}\W*$", re.I)
_SENTENCES = re.compile(r"(?<=[.!?;])\s+|\n+")

# Prior exposure of each asset to each factor's (signed) shock, used for asset_impacts.
_EXPOSURE = {
    "SPY": {"rates": -0.5, "inflation": -0.3, "growth": 0.8, "energy": -0.1,
            "geopolitics": -0.5, "credit": -0.8},
    "TLT": {"rates": -0.9, "inflation": -0.6, "growth": -0.3, "geopolitics": 0.3, "credit": 0.3},
    "GLD": {"rates": -0.3, "inflation": 0.5, "geopolitics": 0.6, "credit": 0.2},
    "USO": {"energy": 1.0, "growth": 0.3, "geopolitics": 0.4},
}  # fmt: skip


def _intensity(sentence: str) -> float:
    if _STRONG.search(sentence):
        return 1.0
    if _MODERATE.search(sentence):
        return 0.4
    return 0.7


@dataclass
class LexiconParser:
    """Deterministic phrase-rule parser. Fast, reproducible, and the SLM's baseline."""

    tickers: list[str] = field(default_factory=lambda: list(_EXPOSURE))
    name: str = "lexicon"
    model: str = "rules-v1"

    def parse(self, text: str) -> RiskSignal:
        sentences = [s for s in _SENTENCES.split(text) if s.strip()]
        raw = dict.fromkeys(FACTORS, 0.0)
        evidence, matched = [], 0
        for sent in sentences:
            hit = False
            for factor, rules in _DIRECTION_RE.items():
                for pattern, sign in rules:
                    for m in pattern.finditer(sent):
                        flip = -1 if _NEGATION.search(sent[: m.start()]) else 1
                        raw[factor] += sign * flip * _intensity(sent)
                        hit = True
            if hit:
                matched += 1
                if len(evidence) < 3:
                    evidence.append(sent.strip()[:160])
        shocks = {k: float(np.tanh(v)) for k, v in raw.items()}

        pos, neg = len(_POSITIVE.findall(text)), len(_NEGATIVE.findall(text))
        sentiment = (pos - neg) / (pos + neg) * min(1.0, (pos + neg) / 3) if pos + neg else 0.0
        net_risk = len(_RISK_UP.findall(text)) - len(_RISK_DOWN.findall(text))
        direction = "up" if net_risk > 0 else "down" if net_risk < 0 else "neutral"

        impacts = {}
        for t in self.tickers:
            expo = _EXPOSURE.get(t)
            val = sum(expo.get(k, 0.0) * v for k, v in shocks.items()) if expo else sentiment
            impacts[t] = float(np.clip(np.tanh(val), -1, 1))
        conf = 0.05 if not sentences else max(0.05, min(1.0, matched / len(sentences)))
        return RiskSignal(
            sentiment_score=float(np.clip(sentiment, -1, 1)),
            macro_shock_vector=ShockVector(**shocks),
            volatility_premium_signal=VolPremiumSignal(
                direction=direction, magnitude=min(1.0, abs(net_risk) / 3)
            ),
            asset_impacts=impacts,
            confidence=float(conf),
            evidence=evidence,
        )


# --------------------------------------------------------------------------------------------
# Ollama backend
# --------------------------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial risk parser. Read the text and return ONLY a JSON object \
matching the given schema. Judge only what the text says; do not use outside knowledge of later \
events. Placeholders like [DATE], [YEAR], [EVENT], [PERSON] hide identifying details on purpose.

Fields:
- sentiment_score: overall tone for risk assets, -1 (very negative) to 1 (very positive).
- macro_shock_vector: one score per factor from -1 to 1; use 0 when the text does not mention \
the factor. Each factor is named by its POSITIVE direction: rate_pressure, inflation_pressure, \
growth_momentum, energy_price_pressure, geopolitical_tension, credit_stress. News that relieves a \
pressure gets a NEGATIVE score: rate cuts or easing make rate_pressure negative, cooling \
inflation makes inflation_pressure negative, a ceasefire makes geopolitical_tension negative, \
tightening spreads or calmer credit markets make credit_stress negative. Keeping rates unchanged \
is 0, not positive.
- volatility_premium_signal: direction "up" if the text implies more market risk or uncertainty, \
"down" if less, else "neutral"; magnitude 0 to 1.
- asset_impacts: expected direction for each listed ticker, -1 to 1.
- confidence: 0 to 1, how clearly the text supports your answer.
- evidence: up to 3 short quotes from the text supporting the answer."""

# Balanced on purpose: one example where every mentioned pressure falls, one where they rise.
FEW_SHOT = [
    ("Crude slumps sharply as supply glut deepens and demand weakens.",
     {"sentiment_score": -0.4,
      "macro_shock_vector": {"rate_pressure": 0, "inflation_pressure": -0.2,
                             "growth_momentum": -0.3, "energy_price_pressure": -0.9,
                             "geopolitical_tension": 0, "credit_stress": 0},
      "volatility_premium_signal": {"direction": "up", "magnitude": 0.3},
      "asset_impacts": {"SPY": -0.1, "TLT": 0.2, "GLD": 0.0, "USO": -0.9},
      "confidence": 0.9, "evidence": ["Crude slumps sharply as supply glut deepens"]}),
    ("The Committee decided to maintain the target range. Economic activity has continued to "
     "expand at a solid pace, and inflation has eased somewhat.",
     {"sentiment_score": 0.4,
      "macro_shock_vector": {"rate_pressure": 0, "inflation_pressure": -0.3,
                             "growth_momentum": 0.5, "energy_price_pressure": 0,
                             "geopolitical_tension": 0, "credit_stress": 0},
      "volatility_premium_signal": {"direction": "down", "magnitude": 0.2},
      "asset_impacts": {"SPY": 0.4, "TLT": 0.1, "GLD": -0.1, "USO": 0.1},
      "confidence": 0.8, "evidence": ["expand at a solid pace", "inflation has eased somewhat"]}),
    ("The Committee decided to lower the target range. Credit spreads have narrowed and funding "
     "markets have calmed, and geopolitical tensions eased after a ceasefire.",
     {"sentiment_score": 0.5,
      "macro_shock_vector": {"rate_pressure": -0.8, "inflation_pressure": 0,
                             "growth_momentum": 0, "energy_price_pressure": 0,
                             "geopolitical_tension": -0.6, "credit_stress": -0.6},
      "volatility_premium_signal": {"direction": "down", "magnitude": 0.5},
      "asset_impacts": {"SPY": 0.5, "TLT": 0.5, "GLD": -0.1, "USO": 0.1},
      "confidence": 0.9, "evidence": ["lower the target range", "Credit spreads have narrowed",
                                      "tensions eased after a ceasefire"]}),
    ("Credit spreads widened sharply amid default worries, and the Committee raised the target "
     "range as inflation remains elevated.",
     {"sentiment_score": -0.6,
      "macro_shock_vector": {"rate_pressure": 0.7, "inflation_pressure": 0.5,
                             "growth_momentum": 0, "energy_price_pressure": 0,
                             "geopolitical_tension": 0, "credit_stress": 0.8},
      "volatility_premium_signal": {"direction": "up", "magnitude": 0.7},
      "asset_impacts": {"SPY": -0.6, "TLT": -0.3, "GLD": 0.2, "USO": -0.1},
      "confidence": 0.9, "evidence": ["Credit spreads widened sharply",
                                      "raised the target range"]}),
]  # fmt: skip


@dataclass
class OllamaParser:
    """Local SLM via Ollama's /api/chat with a JSON schema as ``format`` (constrained decoding)."""

    model: str = "qwen2.5:3b"
    url: str = "http://localhost:11434"
    tickers: list[str] = field(default_factory=lambda: list(_EXPOSURE))
    timeout_s: float = 60.0
    seed: int = 0
    fallback: LexiconParser | None = None
    client: httpx.Client | None = None
    name: str = "ollama"
    fallbacks: int = 0
    retries: int = 0

    def __post_init__(self) -> None:
        self.fallback = self.fallback or LexiconParser(self.tickers)
        self.client = self.client or httpx.Client(base_url=self.url, timeout=self.timeout_s)

    def _messages(self, text: str, error: str | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + f"\nTickers: {self.tickers}"}]
        for example, answer in FEW_SHOT:
            msgs += [{"role": "user", "content": example},
                     {"role": "assistant", "content": json.dumps(answer)}]  # fmt: skip
        msgs.append({"role": "user", "content": text})
        if error:
            msgs.append({"role": "user", "content": f"Your previous answer was invalid: {error}. "
                                                    "Return corrected JSON only."})  # fmt: skip
        return msgs

    def _call(self, text: str, error: str | None = None) -> str:
        resp = self.client.post("/api/chat", json={
            "model": self.model,
            "messages": self._messages(text, error),
            "format": RiskSignal.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0, "seed": self.seed},
        })  # fmt: skip
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def parse(self, text: str) -> RiskSignal:
        error: str | None = None
        for _attempt in range(2):
            try:
                content = self._call(text, error)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                log.warning("Ollama call failed (%s); using lexicon fallback", exc)
                break
            try:
                sig = RiskSignal.model_validate_json(content)
                # Constrained decoding fixes the shape, not the ticker set: normalise it.
                sig.asset_impacts = {t: sig.asset_impacts.get(t, 0.0) for t in self.tickers}
                return sig
            except ValidationError as exc:
                error = str(exc).splitlines()[0]
                self.retries += 1
        self.fallbacks += 1
        return self.fallback.parse(text)

    def ping(self) -> bool:
        """True if the Ollama server answers and has the model pulled."""
        try:
            tags = self.client.get("/api/tags").json()
        except (httpx.HTTPError, ValueError):
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names


# --------------------------------------------------------------------------------------------
# Cache + batch parsing
# --------------------------------------------------------------------------------------------


class Parser(Protocol):
    name: str
    model: str

    def parse(self, text: str) -> RiskSignal: ...


class CachedParser:
    """SQLite-backed cache in front of any parser; also records per-call latency."""

    def __init__(self, inner: Parser, path: Path, prompt_version: str = "v1") -> None:
        self.inner, self.prompt_version = inner, prompt_version
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS signals (key TEXT PRIMARY KEY, backend TEXT, model TEXT, "
            "prompt_version TEXT, signal TEXT, latency_ms REAL, created REAL)"
        )
        self.hits = self.misses = 0
        self.latencies_ms: list[float] = []

    def key(self, text: str) -> str:
        raw = f"{self.inner.name}|{self.inner.model}|{self.prompt_version}|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def parse(self, text: str) -> RiskSignal:
        k = self.key(text)
        row = self.db.execute("SELECT signal FROM signals WHERE key = ?", (k,)).fetchone()
        if row:
            self.hits += 1
            return RiskSignal.model_validate_json(row[0])
        t0 = time.perf_counter()
        sig = self.inner.parse(text)
        ms = (time.perf_counter() - t0) * 1000
        self.misses += 1
        self.latencies_ms.append(ms)
        self.db.execute(
            "INSERT OR REPLACE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?)",
            (k, self.inner.name, self.inner.model, self.prompt_version, sig.model_dump_json(), ms,
             time.time()),
        )  # fmt: skip
        self.db.commit()
        return sig

    def close(self) -> None:
        self.db.close()


def parse_items(parser: Parser, items: Iterable[NewsItem], mask: bool = True) -> pd.DataFrame:
    """Parse news items into one row per item: id, published_at and flattened signal fields."""
    rows = []
    for item in items:
        text = mask_entities(item.text) if mask else item.text
        sig = parser.parse(text)
        rows.append({"id": item.id, "published_at": item.published_at, **sig.flat()})
    return pd.DataFrame(rows)


def build_parser(cfg, tickers: list[str]) -> CachedParser:
    """Construct the configured backend wrapped in the cache. ``cfg`` is a ``ParserConfig``."""
    if cfg.backend == "lexicon":
        inner: Parser = LexiconParser(tickers)
    elif cfg.backend == "ollama":
        inner = OllamaParser(model=cfg.model, url=cfg.ollama_url, tickers=tickers,
                             timeout_s=cfg.timeout_s)  # fmt: skip
        if not inner.ping():
            raise RuntimeError(
                f"Ollama at {cfg.ollama_url} is not reachable or '{cfg.model}' is not pulled. "
                f"Install with: brew install ollama && ollama serve & ollama pull {cfg.model}"
            )
    else:
        raise ValueError(f"unknown parser backend {cfg.backend!r}")
    return CachedParser(inner, cfg.cache_path, cfg.prompt_version)
