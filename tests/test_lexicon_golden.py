import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_lexicon_golden_matches_python():
    # The browser lexicon is tested against this file, so it must track the Python parser.
    spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/make_lexicon_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert json.loads(mod.OUT.read_text()) == mod.golden(), (
        "site/tests/lexicon_golden.json is stale: run `python scripts/make_lexicon_golden.py`"
    )
