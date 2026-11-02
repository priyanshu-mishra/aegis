"""A synthetic market whose news has a *known* information content.

A latent macro-shock process z_t (one series per factor) drives three things:

* **volatility** of next-day returns: σ_{i,t+1} = base_i·(1 + 1.5·regime_t)·exp(c Σ_k V_ik|z_{k,t}|)
* **drift** of next-day returns:      μ_{i,t+1} = 2e-4 + 1e-3 · Σ_k M_ik z_{k,t}
* **news headlines** on day t, whose wording reflects an observation z_{k,t} + noise/snr

Large shocks also trigger a high-volatility **regime** on the day they occur. A volatility index
(named after the first feature ticker, e.g. ``^VIX``) tracks the regime with a lag of a few days
and ignores the shocks themselves, and EWMA only reacts once large returns arrive. So news is an
early warning, by construction: the world is built so that text carries information beyond prices
and the vol index. This is a controlled test of whether the parser can recover a signal that is
known to exist, not a claim about real markets. With ``snr = 0`` headlines are pure noise and the
text layer should add nothing.

Everything the model may see (prices, the vol index, news) is separated from ground truth
(``latent``, ``regime``, ``sigma``, ``news_truth``), which is for evaluation only.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aegis.data.news import NY, NewsItem, NewsStore

FACTORS = ("rates", "inflation", "growth", "energy", "geopolitics", "credit")

# Daily base vol, vol loadings V (on |z|) and drift loadings M (on z), per known ticker.
_BASE_VOL = {"SPY": 0.0062, "TLT": 0.0052, "GLD": 0.0058, "USO": 0.0125}  # ≈ real ann. vols overall
_VOL_LOAD = {
    "SPY": {"growth": 0.6, "credit": 0.8, "geopolitics": 0.5, "rates": 0.3},
    "TLT": {"rates": 0.8, "inflation": 0.6},
    "GLD": {"geopolitics": 0.6, "inflation": 0.5},
    "USO": {"energy": 1.0, "geopolitics": 0.5},
}
_DRIFT_LOAD = {
    "SPY": {"growth": 1.0, "credit": -1.0, "rates": -0.5},
    "TLT": {"rates": -1.0, "inflation": -0.5, "growth": -0.3},
    "GLD": {"geopolitics": 0.6, "inflation": 0.5},
    "USO": {"energy": 1.0, "geopolitics": 0.4},
}
_CORR = {
    ("SPY", "TLT"): -0.3, ("SPY", "GLD"): 0.05, ("SPY", "USO"): 0.35,
    ("TLT", "GLD"): 0.25, ("TLT", "USO"): -0.15, ("GLD", "USO"): 0.2,
}  # fmt: skip

# Headline templates: factor → sign → list of (title, body). {adv} is filled by intensity.
_TEMPLATES = {
    "rates": {
        1: [
            (
                "Fed officials signal {adv} tighter policy as rate hike bets build",
                "Policymakers said further increases may be needed to restrain demand.",
            ),
            (
                "Treasury yields climb {adv} on hawkish central bank remarks",
                "Traders priced in additional tightening over the coming months.",
            ),
        ],
        -1: [
            (
                "Central bank hints at rate cuts; yields fall {adv}",
                "Officials said policy could ease if conditions soften.",
            ),
            (
                "Dovish Fed commentary sends yields {adv} lower",
                "Markets moved to price in an earlier start to easing.",
            ),
        ],
    },
    "inflation": {
        1: [
            (
                "Consumer prices rise {adv} faster than expected, stoking inflation fears",
                "Core measures accelerated, with services costs running hot.",
            ),
            (
                "Producer prices jump {adv} as cost pressures intensify",
                "Input costs rose across manufacturing and transport.",
            ),
        ],
        -1: [
            (
                "Inflation cools {adv} as price pressures ease",
                "Goods prices declined and shelter costs moderated.",
            ),
            (
                "Price growth slows {adv}, easing pressure on policymakers",
                "Economists said disinflation appears to be broadening.",
            ),
        ],
    },
    "growth": {
        1: [
            (
                "Economy expands {adv} on strong hiring and robust consumer spending",
                "Payrolls beat forecasts and retail sales surprised to the upside.",
            ),
            (
                "Manufacturing activity accelerates {adv} as new orders surge",
                "Firms reported rising output and improving demand.",
            ),
        ],
        -1: [
            (
                "Growth slows {adv} as factory orders contract and layoffs rise",
                "Weak demand weighed on output and business confidence.",
            ),
            (
                "Recession worries grow {adv} after disappointing jobs report",
                "Hiring stalled and unemployment claims climbed.",
            ),
        ],
    },
    "energy": {
        1: [
            (
                "Oil prices jump {adv} on supply disruption and output cuts",
                "Producers announced reductions while inventories fell.",
            ),
            (
                "Crude rallies {adv} as pipeline outage tightens supply",
                "Analysts warned of shortages heading into peak demand.",
            ),
        ],
        -1: [
            (
                "Crude slumps {adv} as supply glut deepens and demand weakens",
                "Inventories rose for a third straight week.",
            ),
            (
                "Oil falls {adv} after producers boost output",
                "Traders cited ample supply and softer consumption.",
            ),
        ],
    },
    "geopolitics": {
        1: [
            (
                "Geopolitical tensions escalate {adv}; conflict fears rattle markets",
                "Investors sought safety amid reports of military escalation.",
            ),
            (
                "Sanctions threat intensifies {adv} as diplomatic talks collapse",
                "Officials warned of retaliation and trade disruption.",
            ),
        ],
        -1: [
            (
                "Tensions ease {adv} after ceasefire agreement",
                "Diplomatic progress lifted sentiment across regions.",
            ),
            (
                "Diplomatic breakthrough {adv} reduces conflict risk",
                "Leaders agreed to resume negotiations.",
            ),
        ],
    },
    "credit": {
        1: [
            (
                "Credit spreads widen {adv} amid default worries and bank stress",
                "Lenders tightened standards and funding costs rose.",
            ),
            (
                "Corporate bond selloff deepens {adv} as downgrades mount",
                "Investors pulled money from high-yield funds.",
            ),
        ],
        -1: [
            (
                "Credit markets calm {adv} as spreads tighten",
                "Funding conditions stabilized and issuance resumed.",
            ),
            (
                "Bank stress fades {adv} as deposits stabilize",
                "Regulators said the system remains well capitalized.",
            ),
        ],
    },
}
_NEUTRAL = {
    "rates": "Rate expectations", "inflation": "Price data", "growth": "Economic data",
    "energy": "Oil market", "geopolitics": "Geopolitical situation", "credit": "Credit market",
}  # fmt: skip
_ADVERBS = {"strong": ("sharply", "significantly"), "moderate": ("modestly", "slightly")}


@dataclass
class SyntheticWorld:
    # Observable
    prices: pd.DataFrame
    features: pd.DataFrame
    news: NewsStore
    # Ground truth (evaluation only, never a model input)
    latent: pd.DataFrame  # z_t per factor
    regime: pd.Series  # 0 calm, 1 stress
    sigma: pd.DataFrame  # true σ of each day's return
    news_truth: pd.DataFrame  # per item: factor, z, observed value


def _corr_matrix(tickers: list[str], rng: np.random.Generator) -> np.ndarray:
    n = len(tickers)
    c = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            pair = (tickers[i], tickers[j])
            rho = _CORR.get(pair, _CORR.get(pair[::-1], rng.uniform(-0.2, 0.3)))
            c[i, j] = c[j, i] = rho
    # Repair to positive definite if random fills broke it.
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, 1e-3, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def _loadings(tickers, table, rng, scale) -> np.ndarray:
    out = np.zeros((len(tickers), len(FACTORS)))
    for i, t in enumerate(tickers):
        known = table.get(t)
        for k, f in enumerate(FACTORS):
            out[i, k] = known.get(f, 0.0) if known is not None else rng.uniform(-scale, scale)
    return out


def _headline(rng, factor: str, value: float) -> tuple[str, str]:
    mag = abs(value)
    if mag < 0.15:
        return (f"{_NEUTRAL[factor]} little changed as markets await further signals",
                "Analysts said the outlook was broadly unchanged.")  # fmt: skip
    options = _TEMPLATES[factor][1 if value > 0 else -1]
    title, body = options[rng.integers(len(options))]
    adv = rng.choice(_ADVERBS["strong" if mag >= 0.5 else "moderate"])
    return title.format(adv=adv), body


def generate_world(
    tickers: list[str],
    vol_index_name: str = "^VIX",
    start: dt.date = dt.date(2010, 1, 1),
    n_days: int = 4000,
    snr: float = 0.5,
    news_per_day: float = 1.0,
    seed: int = 0,
    shock_persistence: float = 0.92,
    shock_vol_scale: float = 1.0,
    stress_trigger: float = 1.2,
) -> SyntheticWorld:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days + 1, name="date")
    T, K, N = len(dates), len(FACTORS), len(tickers)

    # Latent shocks: sparse jumps into a persistent AR(1), squashed into (−1, 1).
    x = np.zeros((T, K))
    jumps = (rng.random((T, K)) < 0.02) * rng.normal(0, 0.8, (T, K))
    for t in range(1, T):
        x[t] = shock_persistence * x[t - 1] + jumps[t]
    z = np.tanh(x)

    # Two-state volatility regime. Calm → stress is mostly triggered by a large shock on that
    # day (which the day's news reports); stress decays back to calm at random.
    big_shock = np.abs(jumps).max(axis=1) > stress_trigger
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        if regime[t - 1] == 0:
            p_enter = 0.002 + (0.5 if big_shock[t] else 0.0)
            regime[t] = int(rng.random() < p_enter)
        else:
            regime[t] = int(rng.random() >= 0.04)

    base = np.array([_BASE_VOL.get(t, 0.008) for t in tickers])
    V = np.abs(_loadings(tickers, _VOL_LOAD, rng, 0.6))
    M = _loadings(tickers, _DRIFT_LOAD, rng, 0.6)
    L = np.linalg.cholesky(_corr_matrix(tickers, rng))

    # Return on day t is driven by state known at the close of t−1.
    sigma = np.full((T, N), np.nan)
    sigma[1:] = (
        base * (1 + 1.5 * regime[:-1, None]) * np.exp(shock_vol_scale * np.abs(z[:-1]) @ V.T)
    )
    mu = np.full((T, N), np.nan)
    mu[1:] = 2e-4 + 1e-3 * z[:-1] @ M.T
    df = 5
    eps = rng.standard_t(df, (T, N)) * np.sqrt((df - 2) / df) @ L.T
    r = np.zeros((T, N))
    r[1:] = mu[1:] + sigma[1:] * eps[1:]
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=tickers)

    # Vol index at close t: annualised SPY-like vol from a *lagged* view of the regime
    # (exponentially smoothed, half-life ≈ 3 days). It sees stress only after it starts showing up.
    seen = np.zeros(T)
    for t in range(1, T):
        seen[t] = 0.8 * seen[t - 1] + 0.2 * regime[t]
    vol_base = base[0] * (1 + 1.5 * seen) * np.sqrt(252) * 100
    vol_index = vol_base * np.exp(rng.normal(0, 0.08, T))
    features = pd.DataFrame({vol_index_name: vol_index}, index=dates)

    items, truth = [], []
    for t, day in enumerate(dates):
        weights = np.abs(z[t]) + 0.05
        for j in range(rng.poisson(news_per_day)):
            k = rng.choice(K, p=weights / weights.sum())
            noise = rng.normal()
            observed = z[t, k] + noise / snr if snr > 0 else 0.5 * noise
            observed = float(np.clip(observed, -1, 1))
            title, body = _headline(rng, FACTORS[k], observed)
            minutes = int(rng.integers(6 * 60, 20 * 60))  # 06:00–20:00 New York time
            ts = dt.datetime.combine(day.date(), dt.time(minutes // 60, minutes % 60), NY)
            item_id = f"syn-{day:%Y%m%d}-{j}"
            items.append(NewsItem(item_id, ts, "synthetic", title, body))
            truth.append((item_id, day, FACTORS[k], z[t, k], observed))

    return SyntheticWorld(
        prices=prices,
        features=features,
        news=NewsStore(items),
        latent=pd.DataFrame(z, index=dates, columns=list(FACTORS)),
        regime=pd.Series(regime, index=dates, name="regime"),
        sigma=pd.DataFrame(sigma, index=dates, columns=tickers),
        news_truth=pd.DataFrame(truth, columns=["id", "date", "factor", "z", "observed"]),
    )
