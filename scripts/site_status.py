"""Write site/public/data/status.json: whether this build has fresh data, and as of when.

Usage: python scripts/site_status.py --fresh true|false
"""

import argparse
import datetime as dt
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "site" / "public" / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--fresh", choices=["true", "false"], required=True)
args = parser.parse_args()

meta = json.loads((DATA / "meta.json").read_text())
status = {
    "fresh": args.fresh == "true",
    "checked": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    "data_last_day": meta["data"]["last_day"],
    "exported": meta["exported"],
}
(DATA / "status.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status))
