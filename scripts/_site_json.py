"""Merge one section into site/public/data/extensions.json (written by the extension studies)."""

import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "site" / "public" / "data" / "extensions.json"


def write_section(key: str, payload: dict) -> None:
    data = json.loads(PATH.read_text()) if PATH.exists() else {}
    data[key] = payload
    PATH.write_text(json.dumps(data, indent=1) + "\n")
