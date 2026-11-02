"""Price harvesting (yfinance + parquet cache) and return/volatility transforms.

Timing convention used throughout Aegis: row ``t`` of any *feature* frame may only use data
known at the close of trading day ``t``. Functions that look forward (targets) say so in their
name (``forward_*``) and are never fed into features.
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
TRADING_DAYS = 252


def _cache_path(cache_dir: Path, tickers: list[str], start, end) -> Path:
    key = hashlib.sha256(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:12]
    return Path(cache_dir) / f"prices_{key}.parquet"


def _download(tickers: list[str], start, end, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # yfinance emits pandas deprecation noise
                raw = yf.download(
                    tickers,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
            if missing:
                raise RuntimeError(f"no data returned for {missing}")
            return close[tickers]
        except Exception as exc:  # network errors, rate limits, empty frames
            last_exc = exc
            wait = 2**attempt
            log.warning("yfinance attempt %d failed (%s); retrying in %ss", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"yfinance download failed after {retries} attempts") from last_exc


def fetch_prices(
    tickers: list[str], start, end=None, cache_dir: Path = Path("data"), refresh: bool = False
) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker, on dates where *every* ticker traded.

    Results are cached to parquet; a cached file is used whenever present unless ``refresh``.
    If the download fails and a cache exists, the cache is used with a warning.
    """
    path = _cache_path(cache_dir, tickers, start, end)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        prices = _download(tickers, start, end)
    except RuntimeError:
        if path.exists():
            log.warning("download failed; using cached prices at %s", path)
            return pd.read_parquet(path)
        raise
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices.index.name = "date"
    prices.columns.name = None
    prices = prices.dropna(how="any").sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t / P_{t-1}); the first row is dropped."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def trailing_realized_vol(returns: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame:
    """sqrt(Σ_{i=0}^{window-1} r²_{t-i}): uses returns up to and including day t only."""
    return np.sqrt((returns**2).rolling(window).sum())


def forward_realized_vol(returns: pd.DataFrame | pd.Series, horizon: int):
    """TARGET, looks forward: RV_{t→t+h} = sqrt(Σ_{i=1}^{h} r²_{t+i}). Last h rows are NaN."""
    return np.sqrt((returns**2).rolling(horizon).sum().shift(-horizon))


def portfolio_weights(
    returns: pd.DataFrame,
    scheme: str = "equal",
    custom: dict[str, float] | None = None,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Weights applied to day t's returns, decided at the close of day t−1.

    ``inverse_vol`` uses trailing vol through t−1 (hence the shift), so day t's own return never
    influences the weight it is multiplied by.
    """
    cols = returns.columns
    if scheme == "equal":
        w = pd.DataFrame(1.0 / len(cols), index=returns.index, columns=cols)
    elif scheme == "custom":
        if custom is None:
            raise ValueError("custom scheme needs weights")
        w = pd.DataFrame(
            [[custom[c] for c in cols]] * len(returns), index=returns.index, columns=cols
        )
    elif scheme == "inverse_vol":
        inv = 1.0 / trailing_realized_vol(returns, vol_window).shift(1)
        w = inv.div(inv.sum(axis=1), axis=0)
        w = w.fillna(1.0 / len(cols))  # warm-up period before enough history exists
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    return w


def portfolio_log_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Daily-rebalanced portfolio: R_t = Σ_i w_{i,t} (e^{r_{i,t}} − 1), returned as ln(1 + R_t)."""
    simple = (np.expm1(returns) * weights).sum(axis=1)
    return np.log1p(simple).rename("portfolio")


@dataclass
class MarketData:
    prices: pd.DataFrame  # adjusted close, portfolio tickers
    returns: pd.DataFrame  # log returns, portfolio tickers
    features: pd.DataFrame  # exogenous series observed at close t (e.g. ^VIX level)
    weights: pd.DataFrame  # weights applied to each day's returns
    portfolio: pd.Series  # portfolio log return
    source: str

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        return self.returns.index


def build_market(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    weighting: str,
    custom: dict[str, float] | None,
    source: str,
) -> MarketData:
    returns = log_returns(prices)
    features = features.reindex(returns.index).ffill()
    weights = portfolio_weights(returns, weighting, custom)
    return MarketData(
        prices=prices.loc[returns.index],
        returns=returns,
        features=features,
        weights=weights,
        portfolio=portfolio_log_returns(returns, weights),
        source=source,
    )
