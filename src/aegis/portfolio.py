"""Portfolio application: turning calibrated risk forecasts into position sizes.

Every rule decides an exposure at the close of day t using only information known then, and
earns the return of day t+1. Costs are charged on turnover (changes in exposure and in asset
weights) at a flat rate in basis points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class StrategyResult:
    exposure: pd.Series  # decided at close of t, applied to the return of t+1
    gross: pd.Series  # simple return earned on day t+1, indexed by the decision day t
    net: pd.Series  # after transaction costs
    turnover: pd.Series


def simple_returns(log_returns: pd.Series | pd.DataFrame):
    return np.expm1(log_returns)


def run_exposure(
    asset_simple: pd.Series,
    exposure: pd.Series,
    cost_bps: float = 5.0,
    extra_turnover: pd.Series | None = None,
) -> StrategyResult:
    """Apply exposure_t to the return of t+1. ``asset_simple`` is indexed by return day.

    Turnover_t = |exposure_t − exposure_{t−1}| (+ ``extra_turnover``, e.g. weight changes);
    cost_t = turnover_t · cost_bps / 10⁴, paid at t.
    """
    nxt = asset_simple.shift(-1).reindex(exposure.index)
    gross = exposure * nxt
    turnover = exposure.diff().abs().fillna(exposure.abs())
    if extra_turnover is not None:
        turnover = turnover + extra_turnover.reindex(exposure.index).fillna(0.0)
    net = gross - turnover * cost_bps / 1e4
    return StrategyResult(exposure, gross, net, turnover)


def vol_target_exposure(
    sigma_daily: pd.Series, target_ann: float, scale: float = 1.0, max_leverage: float = 2.0
) -> pd.Series:
    """exposure_t = min(L, scale · target / (σ̂_t · √252)) with σ̂_t a daily vol forecast."""
    raw = scale * target_ann / (sigma_daily * np.sqrt(TRADING_DAYS))
    return raw.clip(upper=max_leverage).rename("exposure")


def calibrate_scale(
    asset_simple: pd.Series, sigma_daily: pd.Series, target_ann: float, max_leverage: float
) -> float:
    """Constant k so the uncapped rule hits ``target_ann`` realized vol on the given (pre-test)
    sample. Each sizing input has its own bias (e.g. log-vol forecasts sit below realized vol),
    so each rule gets its own k, fixed before the test period starts."""
    exp1 = vol_target_exposure(sigma_daily, target_ann, 1.0, max_leverage=np.inf)
    r = run_exposure(asset_simple, exp1, cost_bps=0).gross.dropna()
    realized = r.std() * np.sqrt(TRADING_DAYS)
    return float(target_ann / realized) if realized > 0 else 1.0


def var_limit_exposure(var_level: pd.Series, limit: float, max_leverage: float = 1.0) -> pd.Series:
    """Scale down so the forecast VaR of the position equals the loss limit:
    exposure_t = min(L, limit / VaR_t), VaR_t = −var_level_t > 0."""
    var = -var_level
    return (limit / var).clip(upper=max_leverage).rename("exposure")


# --------------------------------------------------------------------------------------------
# Allocation: equal risk contribution (risk parity)
# --------------------------------------------------------------------------------------------


def erc_weights(cov: np.ndarray, iters: int = 30) -> np.ndarray:
    """Long-only equal-risk-contribution weights for each covariance in ``cov`` (T, N, N).

    Solves the convex problem  min ½ yᵀΣy − (1/N) Σ log y_i  by Newton's method (Spinu, 2013);
    w = y / Σy then has w_i (Σw)_i equal for all i. Rows with NaN covariance give NaN weights.
    """
    T, N = cov.shape[:2]
    out = np.full((T, N), np.nan)
    b = np.full(N, 1.0 / N)
    for t in range(T):
        S = cov[t]
        if not np.isfinite(S).all():
            continue
        y = 1.0 / np.sqrt(np.diag(S))
        for _ in range(iters):
            g = S @ y - b / y
            H = S + np.diag(b / y**2)
            step = np.linalg.solve(H, g)
            # Backtrack to stay in the positive orthant.
            lam = 1.0
            while np.any(y - lam * step <= 0):
                lam *= 0.5
            y = y - lam * step
            if np.abs(g).max() < 1e-12:
                break
        out[t] = y / y.sum()
    return out


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    return inv / inv.sum(axis=1, keepdims=True)


def allocation_returns(
    asset_log: pd.DataFrame, weights: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """Daily-rebalanced portfolio: weights decided at t earn the asset returns of t+1.

    Returns (portfolio simple return indexed by return day, weight turnover indexed by
    decision day)."""
    w_applied = weights.shift(1).reindex(asset_log.index)
    port = (np.expm1(asset_log) * w_applied).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1)
    return port, turnover


# --------------------------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------------------------


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def strategy_metrics(
    res: StrategyResult, target_ann: float, window: int = 21, overshoot: float = 1.25
) -> dict:
    """Annualized return/vol, Sharpe and Sortino (zero risk-free rate), drawdown, Calmar,
    target tracking (share of rolling ``window``-day periods with realized vol above
    ``overshoot`` × target), exposure and cost statistics."""
    r = res.net.dropna()
    ann_ret = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
    downside = float(r[r < 0].std() * np.sqrt(TRADING_DAYS))
    mdd = max_drawdown(r)
    roll = r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    monthly = (1 + r).resample("ME").prod() - 1
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol else np.nan,
        "sortino": ann_ret / downside if downside else np.nan,
        "max_drawdown": mdd,
        "calmar": ann_ret / -mdd if mdd < 0 else np.nan,
        "overshoot_share": float((roll.dropna() > overshoot * target_ann).mean()),
        "worst_21d_vol": float(roll.max()),
        "worst_month": float(monthly.min()),
        "avg_exposure": float(res.exposure.reindex(r.index).mean()),
        "annual_turnover": float(res.turnover.reindex(r.index).sum() * TRADING_DAYS / len(r)),
        "cost_drag": float((res.gross - res.net).reindex(r.index).sum() * TRADING_DAYS / len(r)),
    }
