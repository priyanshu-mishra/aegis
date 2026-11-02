"""Assemble market data and news for a run from an :class:`AegisConfig`."""

from __future__ import annotations

from dataclasses import dataclass

from aegis.config import AegisConfig
from aegis.data.fomc import fetch_fomc_statements
from aegis.data.news import NewsStore
from aegis.data.prices import MarketData, build_market, fetch_prices
from aegis.data.synthetic import SyntheticWorld, generate_world


@dataclass
class Dataset:
    market: MarketData
    news: NewsStore
    truth: SyntheticWorld | None = None  # ground truth, synthetic runs only


def load_dataset(cfg: AegisConfig, refresh: bool = False) -> Dataset:
    d = cfg.data
    weighting, custom = cfg.portfolio.weighting, cfg.portfolio.weights

    if d.source == "synthetic":
        world = generate_world(
            d.tickers,
            vol_index_name=d.feature_tickers[0] if d.feature_tickers else "VOL",
            start=d.start,
            n_days=d.synthetic.n_days,
            snr=d.synthetic.snr,
            news_per_day=d.synthetic.news_per_day,
            seed=cfg.seed,
        )
        market = build_market(world.prices, world.features, weighting, custom, "synthetic")
        news = world.news if d.news_source == "synthetic" else NewsStore()
        return Dataset(market, news, truth=world)

    prices = fetch_prices(d.tickers + d.feature_tickers, d.start, d.end, d.cache_dir, refresh)
    market = build_market(
        prices[d.tickers], prices[d.feature_tickers], weighting, custom, "yfinance"
    )
    if d.news_source == "fomc":
        news = fetch_fomc_statements(
            d.start, d.end, d.cache_dir / "fomc_statements.jsonl", refresh=refresh
        )
    else:
        news = NewsStore()
    return Dataset(market, news)
