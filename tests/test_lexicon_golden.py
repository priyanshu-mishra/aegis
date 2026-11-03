import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _close(a, b):
    # Floats may differ in the last bit across platforms' math libraries; the JS test uses 1e-9 too.
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        return a == pytest.approx(b, abs=1e-9)
    return a == b


def test_committed_lexicon_golden_matches_python():
    # The browser lexicon is tested against this file, so it must track the Python parser.
    spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/make_lexicon_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert _close(json.loads(mod.OUT.read_text()), mod.golden()), (
        "site/tests/lexicon_golden.json is stale: run `python scripts/make_lexicon_golden.py`"
    )
