"""Component 2: quantitative factor engine.

* EWMA covariance (RiskMetrics) as the baseline risk matrix
* a semantic shock overlay that rescales it with text-derived signals
* point models μ̂(X), scale models σ̂(X) and quantile models q̂_lo/q̂_hi(X) for target A
  (log forward realised vol), packaged as a :class:`~aegis.conformal_ncm.Forecast`
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aegis.conformal_ncm import Forecast

# --------------------------------------------------------------------------------------------
# Covariance
# --------------------------------------------------------------------------------------------


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94, burn_in: int = 22) -> np.ndarray:
    """RiskMetrics EWMA covariance path, shape (T, N, N).

    Σ_t is the forecast for day t+1 made at the close of day t:

        Σ_t = λ Σ_{t−1} + (1 − λ) r_t r_tᵀ

    initialised at t = burn_in − 1 with the sample (zero-mean) covariance of r_0 … r_{burn_in−1}.
    Rows before that are NaN.
    """
    r = returns.to_numpy(dtype=float)
    T, N = r.shape
    out = np.full((T, N, N), np.nan)
    if len(r) < burn_in:
        return out
    s = r[:burn_in].T @ r[:burn_in] / burn_in
    out[burn_in - 1] = s
    for t in range(burn_in, T):
        s = lam * s + (1 - lam) * np.outer(r[t], r[t])
        out[t] = s
    return out


def portfolio_vol(cov: np.ndarray, weights: pd.DataFrame | np.ndarray) -> np.ndarray:
    """σ_p,t = sqrt(w_tᵀ Σ_t w_t) for every t."""
    w = np.asarray(weights, dtype=float)
    return np.sqrt(np.einsum("ti,tij,tj->t", w, cov, w))


@dataclass(frozen=True)
class ShockOverlay:
    """Per-asset log-vol model  log σ̃_{i,t} = c_i + β_i log σ_{i,t} + B_i · s_t.

    ``B`` is (N, K); ``beta`` and ``c`` are (N,). The identity overlay has β = 1, c = 0, B = 0.
    """

    B: np.ndarray
    beta: np.ndarray
    c: np.ndarray

    @classmethod
    def identity(cls, n_assets: int, n_signals: int) -> ShockOverlay:
        return cls(np.zeros((n_assets, n_signals)), np.ones(n_assets), np.zeros(n_assets))


def apply_shock_overlay(cov: np.ndarray, signals: np.ndarray, overlay: ShockOverlay) -> np.ndarray:
    """Semantic shock overlay: Σ̃_t = D_t Σ_t D_t with

        D_t = diag( exp( c + (β − 1) log σ_t + B s_t ) ),   σ_t = sqrt(diag Σ_t)

    so that sqrt(diag Σ̃_t) = exp(c + β log σ_t + B s_t). A diagonal rescaling keeps every Σ̃_t
    positive semi-definite and leaves correlations untouched.
    """
    log_sigma = np.log(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)))
    log_d = overlay.c + (overlay.beta - 1) * log_sigma + np.nan_to_num(signals) @ overlay.B.T
    d = np.exp(log_d)  # (T, N)
    return cov * d[:, :, None] * d[:, None, :]


def fit_shock_overlay(
    cov: np.ndarray,
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    train_mask: np.ndarray,
    horizon: int,
    ridge_alpha: float = 1.0,
) -> ShockOverlay:
    """Fit (c, β, B) on training rows only by ridge-regressing, for each asset i,

        log RV_{i,t→t+h}  on  [ log σ_{i,t},  s_t ]

    β is left free because forcing β = 1 (a pure rescaling of EWMA) generalises poorly: EWMA is
    noisy and vol mean-reverts, so the fitted β is typically ≈ 0.5. The intercept also absorbs
    the Jensen gap (E[log RV] < log σ). Passing all-zero signals gives a calibrated-EWMA baseline.
    """
    r = returns.to_numpy(dtype=float)
    mean_sq = pd.DataFrame(r**2).rolling(horizon).mean().shift(-horizon).to_numpy()
    log_rv = np.log(np.maximum(np.sqrt(mean_sq), 1e-4))
    log_sigma = np.log(np.maximum(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), 1e-4))
    S = signals.reindex(returns.index).to_numpy(dtype=float)
    ok = (
        train_mask
        & np.isfinite(log_rv).all(axis=1)
        & np.isfinite(log_sigma).all(axis=1)
        & np.isfinite(S).all(axis=1)
    )
    N, K = r.shape[1], S.shape[1]
    B, beta, c = np.zeros((N, K)), np.ones(N), np.zeros(N)
    for i in range(N):
        F = np.column_stack([log_sigma[ok, i], S[ok]])
        fit = Ridge(alpha=ridge_alpha).fit(F, log_rv[ok, i])
        beta[i], B[i], c[i] = fit.coef_[0], fit.coef_[1:], fit.intercept_
    return ShockOverlay(B, beta, c)


# --------------------------------------------------------------------------------------------
# Point, scale and quantile models for target A
# --------------------------------------------------------------------------------------------

HAR_COLUMNS = ["log_rv_d", "log_rv_w", "log_rv_m"]


def purged_kfold(n: int, k: int, gap: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Contiguous K-fold that drops ``gap`` rows either side of each test fold from training.

    With h-day-ahead targets, rows within h of the test fold share future returns with it;
    purging them keeps out-of-fold residuals honest.
    """
    bounds = np.linspace(0, n, k + 1).astype(int)
    idx = np.arange(n)
    for a, b in itertools.pairwise(bounds):
        train = idx[(idx < a - gap) | (idx >= b + gap)]
        yield train, idx[a:b]


RIDGE_ALPHAS = np.logspace(-2, 4, 13)


def _ridge(alpha: float | None = None, cv=None):
    """Standardised ridge; with ``alpha=None`` the penalty is chosen by ``cv`` (purged folds)."""
    reg = Ridge(alpha=alpha) if alpha is not None else RidgeCV(alphas=RIDGE_ALPHAS, cv=cv)
    return make_pipeline(StandardScaler(), reg)


def _mlp(seed: int, members: int = 5):
    """A small neural net: standardised inputs, one hidden layer of 16 tanh units, an L2 penalty.

    Five nets with different initialisations are averaged, which steadies a model trained on only
    about two thousand noisy rows. Early stopping is off: scikit-learn's version holds out a random
    validation split, which would mix time periods.
    """
    nets = [
        (
            f"m{i}",
            MLPRegressor(
                hidden_layer_sizes=(16,), activation="tanh", alpha=1.0,
                max_iter=800, random_state=seed + i,
            ),
        )
        for i in range(members)
    ]  # fmt: skip
    return make_pipeline(StandardScaler(), VotingRegressor(nets))


def _make_model(kind: str, seed: int, quantile: float | None = None):
    if kind == "lightgbm":
        import lightgbm as lgb

        params = dict(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
        )  # fmt: skip
        if quantile is not None:
            return lgb.LGBMRegressor(objective="quantile", alpha=quantile, **params)
        return lgb.LGBMRegressor(**params)
    if quantile is not None:  # linear quantile regression for har_ols / ridge / mlp
        return QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
    if kind == "har_ols":
        return LinearRegression()
    if kind == "ridge":
        return _ridge(alpha=1.0)
    if kind == "mlp":
        return _mlp(seed)
    raise ValueError(f"unknown point model {kind!r}")


@dataclass
class VolForecaster:
    """Predicts target A and wraps μ̂, σ̂ and quantile bounds into a Forecast.

    * μ̂(X): the chosen point model (``har_ols`` uses only HAR columns, as in Corsi 2009).
      ``ridge`` standardises features and picks its penalty by purged K-fold CV, so that adding
      many weak columns (e.g. text signals) is shrunk instead of overfit.
    * σ̂(X): a ridge model of log |out-of-fold residual|, used by the normalized NCM
    * q̂_lo, q̂_hi: quantile models at α/2 and 1 − α/2, used by CQR
    """

    kind: str = "lightgbm"
    alpha: float = 0.05
    horizon: int = 5
    n_folds: int = 5
    seed: int = 0

    def _cols(self, X: pd.DataFrame) -> list[str]:
        return HAR_COLUMNS if self.kind == "har_ols" else list(X.columns)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> VolForecaster:
        if X.isna().any().any() or y.isna().any():
            raise ValueError("drop NaN rows before fitting")
        self.columns_ = self._cols(X)
        Xa, ya = X[self.columns_].to_numpy(), y.to_numpy()

        folds = list(purged_kfold(len(ya), self.n_folds, gap=self.horizon))
        if self.kind == "ridge":
            self.point_ = _ridge(cv=folds).fit(Xa, ya)
            self.ridge_alpha_ = float(self.point_[-1].alpha_)

            def make():
                return _ridge(self.ridge_alpha_)
        else:
            self.point_ = _make_model(self.kind, self.seed).fit(Xa, ya)

            def make():
                return _make_model(self.kind, self.seed)

        # Out-of-fold residuals: in-sample residuals of a boosted model are far too small.
        oof = np.full(len(ya), np.nan)
        for tr, te in folds:
            oof[te] = make().fit(Xa[tr], ya[tr]).predict(Xa[te])
        abs_res = np.abs(ya - oof)
        self.scale_ = _ridge(cv=folds).fit(X.to_numpy(), np.log(abs_res + 1e-6))

        self.q_lo_ = _make_model(self.kind, self.seed, self.alpha / 2).fit(Xa, ya)
        self.q_hi_ = _make_model(self.kind, self.seed, 1 - self.alpha / 2).fit(Xa, ya)
        self.oof_rmse_ = float(np.sqrt(np.mean((ya - oof) ** 2)))
        return self

    def predict(self, X: pd.DataFrame) -> Forecast:
        Xa = X[self.columns_].to_numpy()
        lo, hi = self.q_lo_.predict(Xa), self.q_hi_.predict(Xa)
        # Quantile models are fit separately and can cross; sort so lo ≤ hi.
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        sigma = np.exp(self.scale_.predict(X.to_numpy()))
        return Forecast(mu=self.point_.predict(Xa), sigma=np.maximum(sigma, 1e-6), lo=lo, hi=hi)


def return_forecast(log_port_vol: pd.Series, mean: float = 0.0) -> Forecast:
    """Target B forecast: μ̂ = constant (daily means are ~unpredictable), σ̂ = engine vol."""
    sigma = np.exp(log_port_vol.to_numpy())
    return Forecast(mu=np.full(len(sigma), mean), sigma=sigma)
