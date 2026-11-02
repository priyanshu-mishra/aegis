"""Timestamped news items and an as-of-safe store.

The feature builder may reach text only through :meth:`NewsStore.as_of` or
:meth:`NewsStore.daily_buckets`. Both enforce the rule that an item counts toward trading day t
only if it was published strictly before the 16:00 New York close of day t.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
UTC = dt.UTC
MARKET_CLOSE = dt.time(16, 0)


def market_close_utc(day: dt.date | pd.Timestamp) -> dt.datetime:
    """16:00 America/New_York on ``day``, in UTC (handles daylight-saving changes)."""
    d = day.date() if isinstance(day, pd.Timestamp) else day
    return dt.datetime.combine(d, MARKET_CLOSE, tzinfo=NY).astimezone(UTC)


@dataclass(frozen=True)
class NewsItem:
    id: str
    published_at: dt.datetime  # timezone-aware
    source: str
    title: str
    body: str

    def __post_init__(self) -> None:
        if self.published_at.tzinfo is None:
            raise ValueError(f"news item {self.id}: published_at must be timezone-aware")
        object.__setattr__(self, "published_at", self.published_at.astimezone(UTC))

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()

    def to_json(self) -> dict:
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        return d

    @classmethod
    def from_json(cls, d: dict) -> NewsItem:
        return cls(**{**d, "published_at": dt.datetime.fromisoformat(d["published_at"])})


class NewsStore:
    """Immutable, time-sorted collection of :class:`NewsItem`."""

    def __init__(self, items: Iterable[NewsItem] = ()) -> None:
        self._items = sorted(items, key=lambda i: (i.published_at, i.id))
        ids = [i.id for i in self._items]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate news ids")
        self._times = np.array([i.published_at.timestamp() for i in self._items])

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def as_of(self, cutoff: dt.datetime) -> list[NewsItem]:
        """All items published strictly before ``cutoff``."""
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        n = int(np.searchsorted(self._times, cutoff.timestamp(), side="left"))
        return self._items[:n]

    def daily_buckets(self, trading_days: Iterable) -> dict[pd.Timestamp, list[NewsItem]]:
        """Assign each item to the first trading day whose close is strictly after publication.

        Day t's bucket holds items with close(t−1) ≤ published_at < close(t); weekend and
        overnight news rolls forward to the next session. Items before the first close go to the
        first day; items at or after the last close are dropped (not yet tradable).
        """
        days = pd.DatetimeIndex(trading_days)
        closes = np.array([market_close_utc(d).timestamp() for d in days])
        buckets: dict[pd.Timestamp, list[NewsItem]] = {d: [] for d in days}
        idx = np.searchsorted(closes, self._times, side="right")
        for item, i in zip(self._items, idx, strict=True):
            if i < len(days):
                buckets[days[i]].append(item)
        return buckets

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for item in self._items:
                f.write(json.dumps(item.to_json()) + "\n")

    @classmethod
    def load(cls, path: Path) -> NewsStore:
        with open(path) as f:
            return cls(NewsItem.from_json(json.loads(line)) for line in f if line.strip())
