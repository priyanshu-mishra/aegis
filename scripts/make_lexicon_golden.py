"""Write site/tests/lexicon_golden.json: the Python lexicon parser's output on fixed headlines.

The site's browser port (site/src/lib/lexicon.ts) must reproduce these values exactly; `npm test`
checks it, and tests/test_lexicon_golden.py fails if this file no longer matches Python.

    python scripts/make_lexicon_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

from aegis.slm_parser import LexiconParser

OUT = Path(__file__).resolve().parents[1] / "site/tests/lexicon_golden.json"
TICKERS = ["SPY", "TLT", "GLD", "USO"]

# One case per kind of rule: rising and falling pressures, negation, risk words, and no match.
TEXTS = [
    "The Fed signaled further increases in the target range as inflation remains elevated.",
    "Oil prices surged after the conflict escalated; spreads widened sharply.",
    "Inflation has eased and the Committee decided to lower the target range. "
    "Job gains have been solid.",
    "Growth is not weakening, and credit markets stabilized after a brief selloff.",
    "Ceasefire talks resume negotiations; tensions ease and crude prices fell modestly.",
    "Layoffs mount as manufacturing contracts; the economy is in recession.",
    "Nothing happened today.",
]


def golden() -> list[dict]:
    parser = LexiconParser(TICKERS)
    out = []
    for text in TEXTS:
        s = parser.parse(text)
        out.append({
            "text": text,
            "shocks": s.macro_shock_vector.model_dump(),
            "sentiment": s.sentiment_score,
            "vol": s.volatility_premium_signal.direction,
            "impacts": s.asset_impacts,
            "evidence": s.evidence,
        })  # fmt: skip
    return out


if __name__ == "__main__":
    OUT.write_text(json.dumps(golden()))
    print(f"wrote {OUT}")
