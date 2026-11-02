"""Real text source: FOMC policy statements from federalreserve.gov.

Statement links are discovered from the Fed's meeting calendar pages, then each statement page is
downloaded once and cached as JSON lines. Release times are parsed from the page ("For release at
2:00 p.m. EST"); pages that say only "For immediate release" (pre-2013) get 14:15 ET, the
standard release time in that era. Unscheduled statements (e.g. 15 March 2020) are included.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import httpx

from aegis.data.news import NY, NewsItem, NewsStore

log = logging.getLogger(__name__)

BASE = "https://www.federalreserve.gov"
CURRENT_CALENDAR = f"{BASE}/monetarypolicy/fomccalendars.htm"
HISTORICAL_CALENDAR = f"{BASE}/monetarypolicy/fomchistorical{{year}}.htm"
STATEMENT_URL = f"{BASE}/newsevents/pressreleases/monetary{{date}}a.htm"
DEFAULT_RELEASE = dt.time(14, 15)

# Matches both the current and the legacy (redirecting) statement link formats.
_LINK_RE = re.compile(r'href="[^"]*(?:pressreleases/monetary|press/monetary/)(\d{8})a\.htm"'
                      r"[^>]*>\s*(?:<[^>]+>\s*)*Statement\s*<", re.IGNORECASE)  # fmt: skip
# Current calendar: a "Statement:" label followed by "PDF | HTML" links.
_LABEL_RE = re.compile(r"Statement:\s*</strong>.{0,400}?pressreleases/monetary(\d{8})a\.htm",
                       re.IGNORECASE | re.DOTALL)  # fmt: skip
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?", re.IGNORECASE)
_SKIP_PREFIXES = ("for media inquiries", "voting for", "voting against", "implementation note")


def statement_dates_from_calendar(html: str) -> list[dt.date]:
    """Dates of every link labelled "Statement" on a Fed calendar page."""
    raw = _LINK_RE.findall(html) + _LABEL_RE.findall(html)
    dates = {dt.datetime.strptime(d, "%Y%m%d").date() for d in raw}
    return sorted(dates)


class _ArticleParser(HTMLParser):
    """Collects paragraph text inside ``<div id="article">`` plus the release-time line."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0  # div nesting depth inside the article (0 = outside)
        self.in_p = False
        self.p_class: str | None = None
        self.buf: list[str] = []
        self.paragraphs: list[str] = []
        self.release_line = ""
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div":
            if self.depth:
                self.depth += 1
            elif a.get("id") == "article":
                self.depth = 1
        elif tag == "p" and self.depth:
            if self.in_p:  # previous <p> was never closed
                self._flush()
            self.in_p, self.p_class, self.buf = True, a.get("class"), []
        elif tag == "h3" and self.depth and "title" in (a.get("class") or ""):
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "div" and self.depth:
            self.depth -= 1
        elif tag == "p" and self.in_p:
            self._flush()
        elif tag == "h3":
            self._in_title = False

    def handle_data(self, data):
        if self.in_p:
            self.buf.append(data)
        elif self._in_title:
            self.title += data

    def _flush(self):
        text = " ".join(unescape("".join(self.buf)).split())
        if self.p_class == "releaseTime":
            self.release_line = text
        elif self.p_class is None and text:
            self.paragraphs.append(text)
        self.in_p = False

    def close(self):
        super().close()
        if self.in_p:  # releaseTime <p> is often left unclosed
            self._flush()


def parse_statement(html: str, day: dt.date) -> NewsItem:
    p = _ArticleParser()
    p.feed(html)
    p.close()
    body = [x for x in p.paragraphs if not x.lower().startswith(_SKIP_PREFIXES)]
    if not body:
        raise ValueError(f"no statement text found for {day}")
    m = _TIME_RE.search(p.release_line)
    if m:
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        release = dt.time(hour, int(m.group(2)))
    else:
        release = DEFAULT_RELEASE
    return NewsItem(
        id=f"fomc-{day:%Y%m%d}",
        published_at=dt.datetime.combine(day, release, NY),
        source="fomc",
        title=" ".join(p.title.split()) or "FOMC statement",
        body="\n\n".join(body),
    )


def _get(client: httpx.Client, url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPError as exc:
            if attempt == retries - 1:
                raise
            log.warning("GET %s failed (%s); retrying", url, exc)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_fomc_statements(
    start: dt.date,
    end: dt.date | None = None,
    cache_path: Path = Path("data/fomc_statements.jsonl"),
    refresh: bool = False,
    delay_s: float = 0.3,
) -> NewsStore:
    """All FOMC statements in [start, end], from cache where possible.

    The cache is incremental: only statements not already cached are downloaded.
    """
    end = end or dt.date.today()
    cached = {} if refresh or not cache_path.exists() else {
        i.id: i for i in NewsStore.load(cache_path)
    }  # fmt: skip

    headers = {"User-Agent": "aegis-research/0.1 (FOMC statement loader)"}
    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        dates: set[dt.date] = set()
        current = _get(client, CURRENT_CALENDAR)
        dates.update(statement_dates_from_calendar(current))
        covered_years = {d.year for d in dates}
        for year in range(start.year, end.year + 1):
            if year in covered_years:
                continue
            try:
                html = _get(client, HISTORICAL_CALENDAR.format(year=year))
            except httpx.HTTPStatusError:
                log.warning("no historical calendar for %s", year)
                continue
            dates.update(statement_dates_from_calendar(html))

        wanted = sorted(d for d in dates if start <= d <= end)
        new = 0
        for d in wanted:
            if f"fomc-{d:%Y%m%d}" in cached:
                continue
            html = _get(client, STATEMENT_URL.format(date=f"{d:%Y%m%d}"))
            item = parse_statement(html, d)
            cached[item.id] = item
            new += 1
            time.sleep(delay_s)  # be polite to the Fed's servers

    store = NewsStore(i for i in cached.values() if start <= i.published_at.date() <= end)
    if new or not cache_path.exists():
        NewsStore(cached.values()).save(cache_path)
    log.info("FOMC: %d statements (%d newly downloaded)", len(store), new)
    return store
