import datetime as dt

import numpy as np
import pandas as pd
import pytest

from aegis.config import load_config
from aegis.data import load_dataset
from aegis.data import prices as prices_mod
from aegis.data.fomc import parse_statement, statement_dates_from_calendar
from aegis.data.news import NewsItem, NewsStore, market_close_utc
from aegis.data.prices import log_returns, portfolio_log_returns, portfolio_weights
from aegis.data.synthetic import FACTORS, generate_world

# ---------------------------------------------------------------------------------------------
# News primitives
# ---------------------------------------------------------------------------------------------


def test_market_close_handles_dst():
    assert market_close_utc(dt.date(2024, 3, 8)).hour == 21  # EST, UTC−5
    assert market_close_utc(dt.date(2024, 3, 11)).hour == 20  # EDT, UTC−4


def test_news_item_requires_timezone_and_roundtrips(tmp_path):
    with pytest.raises(ValueError):
        NewsItem("x", dt.datetime(2024, 1, 1, 9), "s", "t", "b")
    item = NewsItem("x", dt.datetime(2024, 1, 1, 9, tzinfo=dt.UTC), "s", "t", "b")
    store = NewsStore([item])
    store.save(tmp_path / "n.jsonl")
    assert list(NewsStore.load(tmp_path / "n.jsonl")) == [item]
    with pytest.raises(ValueError):
        NewsStore([item, item])


# ---------------------------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------------------------


def test_returns_and_equal_weight_portfolio():
    idx = pd.bdate_range("2024-01-01", periods=3)
    px = pd.DataFrame({"A": [100, 110, 99], "B": [50, 50, 55]}, index=idx, dtype=float)
    r = log_returns(px)
    assert r["A"].iloc[0] == pytest.approx(np.log(1.1))
    port = portfolio_log_returns(r, portfolio_weights(r, "equal"))
    assert port.iloc[0] == pytest.approx(np.log1p(0.5 * 0.10 + 0.5 * 0.0))
    assert port.iloc[1] == pytest.approx(np.log1p(0.5 * -0.10 + 0.5 * 0.10))


def test_custom_weights_applied():
    r = pd.DataFrame({"A": [0.01], "B": [0.02]}, index=pd.bdate_range("2024-01-01", periods=1))
    w = portfolio_weights(r, "custom", {"A": 0.25, "B": 0.75})
    assert w.iloc[0].tolist() == [0.25, 0.75]


def test_fetch_prices_caches_and_falls_back(tmp_path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=4, tz="America/New_York")
    frame = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0], "B": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    calls = []

    def fake_download(tickers, start, end, retries=3):
        calls.append(tickers)
        return frame

    monkeypatch.setattr(prices_mod, "_download", fake_download)
    p1 = prices_mod.fetch_prices(["A", "B"], "2024-01-01", cache_dir=tmp_path)
    assert len(p1) == 3 and p1.index.tz is None  # NaN row dropped, tz stripped
    p2 = prices_mod.fetch_prices(["A", "B"], "2024-01-01", cache_dir=tmp_path)
    pd.testing.assert_frame_equal(p1, p2, check_freq=False)
    assert len(calls) == 1  # second call served from cache

    def failing(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(prices_mod, "_download", failing)
    p3 = prices_mod.fetch_prices(["A", "B"], "2024-01-01", cache_dir=tmp_path, refresh=True)
    pd.testing.assert_frame_equal(p1, p3, check_freq=False)
    with pytest.raises(RuntimeError):
        prices_mod.fetch_prices(["C"], "2024-01-01", cache_dir=tmp_path)


# ---------------------------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------------------------

TICKERS = ["SPY", "TLT", "GLD", "USO"]


def test_synthetic_is_deterministic_and_well_formed():
    a = generate_world(TICKERS, n_days=600, seed=3)
    b = generate_world(TICKERS, n_days=600, seed=3)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    assert [i.id for i in a.news] == [i.id for i in b.news]
    assert a.prices.shape == (601, 4) and not a.prices.isna().any().any()
    assert list(a.latent.columns) == list(FACTORS)
    assert a.latent.abs().max().max() < 1
    assert len(a.news) == len(a.news_truth)


def test_synthetic_works_for_unknown_tickers():
    w = generate_world(["AAA", "BBB"], n_days=400, seed=1)
    assert w.prices.shape == (401, 2)


def test_synthetic_news_signal_scales_with_snr():
    def corr(snr):
        w = generate_world(TICKERS, n_days=2000, snr=snr, news_per_day=2, seed=5)
        return np.corrcoef(w.news_truth["z"], w.news_truth["observed"])[0, 1]

    assert corr(2.0) > 0.5
    assert abs(corr(0.0)) < 0.1


def test_synthetic_shocks_drive_future_vol_beyond_vol_index():
    w = generate_world(TICKERS, n_days=4000, seed=11)
    r = log_returns(w.prices)
    future_rv = np.sqrt((r["USO"] ** 2).rolling(5).sum().shift(-5))
    energy = w.latent["energy"].abs().reindex(r.index)
    ok = future_rv.notna()
    assert np.corrcoef(energy[ok], future_rv[ok])[0, 1] > 0.2
    # The vol index tracks the regime, not shocks: near-zero correlation with |z_energy|.
    vix = w.features.iloc[:, 0].reindex(r.index)
    assert abs(np.corrcoef(energy[ok], vix[ok])[0, 1]) < 0.15
    assert (
        vix[w.regime.reindex(r.index) == 1].mean() > 2 * vix[w.regime.reindex(r.index) == 0].mean()
    )


def test_load_dataset_synthetic_offline():
    cfg = load_config(None, ["data.source=synthetic", "data.news_source=synthetic",
                             "data.synthetic.n_days=500"])  # fmt: skip
    ds = load_dataset(cfg)
    assert ds.truth is not None and ds.market.source == "synthetic"
    assert len(ds.market.returns) == 500 and len(ds.news) > 0
    assert ds.market.portfolio.notna().all()


def test_config_rejects_mismatched_news_source():
    with pytest.raises(ValueError, match="news_source"):
        load_config(None, ["data.source=synthetic"])  # default news_source is fomc


# ---------------------------------------------------------------------------------------------
# FOMC parsing (offline fixtures that mimic the two Fed page layouts)
# ---------------------------------------------------------------------------------------------

_CURRENT_CAL = """
<div><strong>Statement:</strong><br>
<a href="/monetarypolicy/files/monetary20230201a1.pdf">PDF</a> |
<a href="/newsevents/pressreleases/monetary20230201a.htm">HTML</a><br>
<a href="/newsevents/pressreleases/monetary20230201a1.htm">Implementation Note</a></div>
"""
_HIST_CAL = """
<p><a href="/newsevents/press/monetary/20100127a.htm">Statement</a></p>
<p><a href="/newsevents/pressreleases/monetary20150318a.htm">Statement</a></p>
<p><a href="/newsevents/pressreleases/monetary20150318b.htm">Other release</a></p>
<p><a href="/monetarypolicy/fomcminutes20150318.htm">Minutes</a></p>
"""
_STATEMENT_NEW = """<html><body><div id="article">
<div class="heading"><p class="article__time">March 01, 2099</p>
<h3 class="title">Test FOMC statement</h3>
<p class="releaseTime">For release at 2:00 p.m. EST
<ul><li><a href="#">Share</a></li></ul></div>
<div class="col"><p>First test paragraph about policy.</p>
<p>Second test paragraph &amp; more.</p>
<p>Voting for the action were: Test Person.</p>
<p>For media inquiries, call 000.</p></div></div></body></html>"""
_STATEMENT_OLD = """<html><body><div id="article">
<p class="releaseTime">For immediate release</p>
<P>Old-format test paragraph.</P></div>
<div><p>Footer text outside the article.</p></div></body></html>"""


def test_calendar_link_discovery():
    assert statement_dates_from_calendar(_CURRENT_CAL) == [dt.date(2023, 2, 1)]
    assert statement_dates_from_calendar(_HIST_CAL) == [dt.date(2010, 1, 27), dt.date(2015, 3, 18)]


def test_parse_statement_new_and_old_layouts():
    new = parse_statement(_STATEMENT_NEW, dt.date(2099, 3, 2))
    assert new.title == "Test FOMC statement"
    assert new.body == "First test paragraph about policy.\n\nSecond test paragraph & more."
    assert new.published_at == dt.datetime(2099, 3, 2, 19, 0, tzinfo=dt.UTC)  # 14:00 EST
    old = parse_statement(_STATEMENT_OLD, dt.date(2010, 6, 23))
    assert old.body == "Old-format test paragraph."
    assert old.published_at == dt.datetime(2010, 6, 23, 18, 15, tzinfo=dt.UTC)  # 14:15 EDT
    with pytest.raises(ValueError):
        parse_statement("<html></html>", dt.date(2020, 1, 1))


@pytest.mark.network
def test_live_fomc_and_prices(tmp_path):
    from aegis.data.fomc import fetch_fomc_statements

    store = fetch_fomc_statements(dt.date(2023, 1, 1), dt.date(2023, 12, 31), tmp_path / "f.jsonl")
    assert len(store) == 8
    px = prices_mod.fetch_prices(["SPY", "^VIX"], "2024-01-01", "2024-02-01", cache_dir=tmp_path)
    assert len(px) > 15


def test_synthetic_stress_follows_shocks_and_vol_index_lags():
    w = generate_world(TICKERS, n_days=4000, seed=11)
    reg = w.regime.to_numpy()
    entries = np.flatnonzero((reg[1:] == 1) & (reg[:-1] == 0)) + 1
    entries = entries[entries < len(reg) - 5]
    assert len(entries) > 10
    # Most stress spells start on a day with a large latent move (which that day's news reports).
    dz = np.abs(np.diff(w.latent.to_numpy(), axis=0)).max(axis=1)
    assert np.mean(dz[entries - 1] > 0.5) > 0.6
    # The vol index only catches up over the following days.
    vix = w.features.iloc[:, 0].to_numpy()
    assert np.median(vix[entries + 5] / vix[entries]) > 1.3
