"""Leakage guards: perturbing anything after time t must not change what is known at t."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from aegis.data.news import NY, NewsItem, NewsStore, market_close_utc
from aegis.data.prices import (
    build_market,
    forward_realized_vol,
    portfolio_weights,
    trailing_realized_vol,
)


def _returns(n=300, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx, columns=["A", "B", "C"])


def _perturb_after(df, t):
    out = df.copy()
    out.iloc[t + 1 :] = out.iloc[t + 1 :] * 5 + 0.03
    return out


@pytest.mark.parametrize("t", [80, 150, 298])
def test_trailing_vol_ignores_future(t):
    r = _returns()
    a = trailing_realized_vol(r, 22).iloc[: t + 1]
    b = trailing_realized_vol(_perturb_after(r, t), 22).iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("t", [80, 150, 298])
def test_inverse_vol_weights_ignore_same_day_and_future(t):
    r = _returns()
    shocked = r.copy()
    shocked.iloc[t:] = shocked.iloc[t:] * 5  # includes day t itself
    a = portfolio_weights(r, "inverse_vol").iloc[: t + 1]
    b = portfolio_weights(shocked, "inverse_vol").iloc[: t + 1]
    pd.testing.assert_frame_equal(a, b)


def test_market_features_ignore_future():
    r = _returns(n=200)
    prices = 100 * np.exp(r.cumsum())
    feats = pd.DataFrame({"VIX": np.linspace(10, 30, 200)}, index=r.index)
    feats.iloc[50:60] = np.nan  # gaps are forward-filled, never back-filled
    t = 100
    a = build_market(prices, feats, "equal", None, "test")
    b = build_market(prices, _perturb_after(feats, t), "equal", None, "test")
    pd.testing.assert_frame_equal(a.features.loc[: r.index[t]], b.features.loc[: r.index[t]])
    assert a.features.loc[r.index[55], "VIX"] == feats.iloc[49]["VIX"]


def test_forward_vol_is_explicitly_forward():
    r = _returns(n=30)["A"]
    h = 5
    fv = forward_realized_vol(r, h)
    for t in range(len(r) - h):
        assert fv.iloc[t] == pytest.approx(np.sqrt((r.iloc[t + 1 : t + 1 + h] ** 2).sum()))
    assert fv.iloc[-h:].isna().all()


# ---------------------------------------------------------------------------------------------
# News timing
# ---------------------------------------------------------------------------------------------


def _item(i, when):
    return NewsItem(f"n{i}", when, "test", f"title {i}", "body")


def test_as_of_is_strict():
    close = market_close_utc(dt.date(2024, 3, 6))
    store = NewsStore(
        [
            _item(0, close - dt.timedelta(seconds=1)),
            _item(1, close),
            _item(2, close + dt.timedelta(hours=1)),
        ]
    )
    assert [i.id for i in store.as_of(close)] == ["n0"]
    with pytest.raises(ValueError):
        store.as_of(dt.datetime(2024, 3, 6, 16))  # naive cutoff rejected


def test_daily_buckets_respect_close():
    days = pd.bdate_range("2024-03-07", "2024-03-12")  # Thu, Fri, Mon, Tue
    thu, fri, mon = days[0], days[1], days[2]
    c_thu = market_close_utc(thu)
    store = NewsStore(
        [
            _item(0, c_thu - dt.timedelta(seconds=1)),  # before Thu close → Thu
            _item(1, c_thu),  # exactly at close → Fri
            _item(2, dt.datetime(2024, 3, 9, 12, tzinfo=NY)),  # Saturday → Mon
            _item(3, dt.datetime(2024, 1, 1, 12, tzinfo=NY)),  # before range → first day
            _item(4, market_close_utc(days[-1])),  # at last close → not yet tradable
        ]
    )
    b = store.daily_buckets(days)
    ids = {d: [i.id for i in items] for d, items in b.items()}
    assert ids[thu] == ["n3", "n0"]
    assert ids[fri] == ["n1"]
    assert ids[mon] == ["n2"]
    assert all("n4" not in v for v in ids.values())


def test_adding_future_news_leaves_past_buckets_unchanged():
    days = pd.bdate_range("2024-01-01", periods=20)
    rng = np.random.default_rng(0)
    base = [
        _item(i, market_close_utc(days[rng.integers(20)]) - dt.timedelta(hours=3))
        for i in range(30)
    ]
    t = days[10]
    later = [_item(100 + i, market_close_utc(t) + dt.timedelta(hours=i)) for i in range(5)]
    a, b = NewsStore(base).daily_buckets(days), NewsStore(base + later).daily_buckets(days)
    for d in days[: 10 + 1]:
        assert [i.id for i in a[d]] == [i.id for i in b[d]]
