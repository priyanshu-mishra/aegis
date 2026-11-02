"""Component 3: conformal prediction via non-conformity measures (NCMs).

Notation
--------
Point model forecasts for a target ``y`` are bundled in a :class:`Forecast` (``mu``, and
optionally ``sigma`` or quantile bounds ``lo``/``hi``). An NCM maps a (forecast, outcome) pair to
a score ``S`` where larger means "less conforming". Given calibration scores ``S_1..S_n`` and an
error level ``α``, the split-conformal threshold is

    q̂ = the k-th smallest S_i,   k = ⌈(n+1)(1−α)⌉                                   (1)

equivalently ``Quantile(S, ⌈(n+1)(1−α)⌉ / n)``. The prediction set for a new point is
``C(X) = {y : S(X, y) ≤ q̂}``, which for each NCM below has a closed-form interval.

Guarantee and its limits
------------------------
If calibration and test points are exchangeable, split conformal satisfies

    1 − α  ≤  P(y_new ∈ C(X_new))  ≤  1 − α + 1/(n+1)                                (2)

(the upper bound needs continuous scores). Coverage is *marginal* (averaged over X), not
conditional on X. Market data are not exchangeable, so :class:`RollingConformal` and
:class:`AdaptiveConformal` are provided for drifting distributions; ACI carries a deterministic
long-run coverage bound that needs no distributional assumption at all (see its docstring).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, fields

import numpy as np

from aegis.metrics import is_covered

Interval = tuple[np.ndarray, np.ndarray]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Conformal threshold q̂ of eq. (1): the ⌈(n+1)(1−α)⌉-th smallest score.

    Returns ``+inf`` when k > n (calibration set too small for this α: the only valid set is
    everything). For ACI, α may leave (0, 1): ``α ≤ 0`` → ``+inf`` (cover everything) and
    ``α ≥ 1`` → ``-inf`` (empty set).
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    if alpha <= 0:
        return math.inf
    if alpha >= 1:
        return -math.inf
    # k = ⌈(n+1)(1−α)⌉. The epsilon stops float noise turning e.g. 20·0.95 = 19.000000000000004
    # into k = 20, which would silently make the interval wider than the theory requires.
    k = math.ceil((n + 1) * (1 - alpha) - 1e-9)
    if k > n:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])


@dataclass(frozen=True)
class Forecast:
    """Point-model outputs for a batch of targets. Unused fields stay ``None``."""

    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    lo: np.ndarray | None = None  # lower quantile forecast (CQR)
    hi: np.ndarray | None = None  # upper quantile forecast (CQR)
    group: np.ndarray | None = None  # regime label per point (Mondrian conformal)

    def __post_init__(self) -> None:
        lengths = set()
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                arr = np.atleast_1d(np.asarray(v, dtype=float))
                object.__setattr__(self, f.name, arr)
                lengths.add(arr.shape[0])
        if not lengths:
            raise ValueError("Forecast needs at least one field")
        if len(lengths) > 1:
            raise ValueError(f"Forecast fields have mismatched lengths: {sorted(lengths)}")
        if self.sigma is not None and not np.all(self.sigma > 0):
            raise ValueError("sigma must be strictly positive")

    def __len__(self) -> int:
        return next(len(v) for f in fields(self) if (v := getattr(self, f.name)) is not None)

    def __getitem__(self, idx: slice) -> Forecast:
        return Forecast(
            **{
                f.name: None if (v := getattr(self, f.name)) is None else v[idx]
                for f in fields(self)
            }
        )


# --------------------------------------------------------------------------------------------
# Non-conformity measures
# --------------------------------------------------------------------------------------------


class NonConformity(ABC):
    """A score function S(X, y) together with the interval form of {y : S(X, y) ≤ q}."""

    name: str
    requires: tuple[str, ...]

    def check(self, f: Forecast) -> None:
        missing = [r for r in self.requires if getattr(f, r) is None]
        if missing:
            raise ValueError(f"{self.name} needs forecast fields {missing}")

    @abstractmethod
    def score(self, f: Forecast, y: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def interval(self, f: Forecast, q: float) -> Interval: ...


class AbsoluteResidual(NonConformity):
    """S = |y − μ̂(X)|  →  C(X) = [μ̂ − q̂, μ̂ + q̂]. Constant width (the PRD's method)."""

    name, requires = "absolute", ("mu",)

    def score(self, f, y):
        return np.abs(y - f.mu)

    def interval(self, f, q):
        return f.mu - q, f.mu + q


class NormalizedResidual(NonConformity):
    """S = |y − μ̂(X)| / σ̂(X)  →  C(X) = [μ̂ − q̂·σ̂, μ̂ + q̂·σ̂]. Width scales with forecast vol."""

    name, requires = "normalized", ("mu", "sigma")

    def score(self, f, y):
        return np.abs(y - f.mu) / f.sigma

    def interval(self, f, q):
        return f.mu - q * f.sigma, f.mu + q * f.sigma


class QuantileResidual(NonConformity):
    """CQR (Romano, Patterson & Candès 2019).

    S = max(q̂_lo(X) − y, y − q̂_hi(X))  →  C(X) = [q̂_lo − q̂, q̂_hi + q̂].
    Negative scores mean y fell inside the quantile band, so q̂ can also *shrink* it.
    """

    name, requires = "cqr", ("lo", "hi")

    def score(self, f, y):
        return np.maximum(f.lo - y, y - f.hi)

    def interval(self, f, q):
        return f.lo - q, f.hi + q


class LowerTailResidual(NonConformity):
    """One-sided score for conformal VaR on returns.

    S = (μ̂ − y) / σ̂  →  C(X) = [μ̂ − q̂·σ̂, +∞). The lower bound is the (1−α) VaR threshold:
    P(y ≥ μ̂ − q̂σ̂) ≥ 1 − α, i.e. VaR_α = −(μ̂ − q̂σ̂) is breached with probability ≤ α.
    """

    name, requires = "lower_tail", ("mu", "sigma")

    def score(self, f, y):
        return (f.mu - y) / f.sigma

    def interval(self, f, q):
        return f.mu - q * f.sigma, np.full_like(f.mu, np.inf)


# --------------------------------------------------------------------------------------------
# Conformal calibrators
# --------------------------------------------------------------------------------------------


class ConformalMethod(ABC):
    """Common interface: ``fit`` on calibration data, ``predict`` intervals, ``update`` online."""

    def __init__(self, ncm: NonConformity, alpha: float) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.ncm, self.alpha = ncm, alpha

    def _scores(self, f: Forecast, y: np.ndarray) -> np.ndarray:
        self.ncm.check(f)
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if len(y) != len(f):
            raise ValueError("forecast and outcome lengths differ")
        return self.ncm.score(f, y)

    @abstractmethod
    def fit(self, f_cal: Forecast, y_cal: np.ndarray) -> ConformalMethod: ...

    @property
    @abstractmethod
    def q(self) -> float: ...

    def predict(self, f: Forecast) -> Interval:
        self.ncm.check(f)
        return self.ncm.interval(f, self.q)

    def update(self, f: Forecast, y: np.ndarray, covered: np.ndarray) -> None:  # noqa: B027
        """Feed back realised outcomes. Static split conformal ignores them."""

    @property
    @abstractmethod
    def scores(self) -> np.ndarray:
        """Calibration scores currently in use."""

    def tail_mean(self) -> float:
        """Mean of the current scores at or beyond q̂: the conformal analogue of Expected
        Shortfall for a one-sided score (an empirical estimate, not a guaranteed quantity)."""
        q = self.q
        if not np.isfinite(q):
            return q
        s = self.scores
        return float(s[s >= q].mean())


class SplitConformal(ConformalMethod):
    """Inductive (split) conformal: one fixed q̂ from the calibration set, eq. (1)."""

    def fit(self, f_cal, y_cal):
        # S_i = NCM(X_i, y_i) for i ∈ D_cal, then q̂ = ⌈(n+1)(1−α)⌉-th smallest.
        self._s = self._scores(f_cal, y_cal)
        self._q = conformal_quantile(self._s, self.alpha)
        return self

    @property
    def q(self):
        return self._q

    @property
    def scores(self):
        return self._s


class RollingConformal(ConformalMethod):
    """Split conformal recomputed on the most recent ``window`` scores as outcomes arrive.

    No finite-sample guarantee under drift, but tracks regime changes with a lag of ~window.
    """

    def __init__(self, ncm: NonConformity, alpha: float, window: int = 500) -> None:
        super().__init__(ncm, alpha)
        self.window = window

    def fit(self, f_cal, y_cal):
        self._buf: deque[float] = deque(self._scores(f_cal, y_cal), maxlen=self.window)
        self._q_cache: float | None = None
        return self

    @property
    def q(self):
        if self._q_cache is None:
            self._q_cache = conformal_quantile(np.fromiter(self._buf, float), self.alpha)
        return self._q_cache

    def update(self, f, y, covered):
        self._buf.extend(self._scores(f, y))
        self._q_cache = None

    @property
    def scores(self):
        return np.fromiter(self._buf, float)


class AdaptiveConformal(ConformalMethod):
    """Adaptive Conformal Inference (Gibbs & Candès, 2021).

    Runs split conformal at a moving level α_t, updated after each realised outcome by

        α_{t+1} = α_t + γ (α − err_t),    err_t = 1{y_t ∉ C_t}                       (3)

    Misses push α_t down (wider sets); hits push it up. For *any* sequence, with one-step
    feedback, the long-run miss rate satisfies

        | (1/T) Σ err_t − α |  ≤  (max(α_1, 1 − α_1) + γ) / (γ T)                   (4)

    No exchangeability is assumed. Larger γ adapts faster but makes the interval width jumpier.

    ``alpha_floor`` (optional) computes the quantile at max(α_t, floor, 1/(n+1)) while α_t
    itself keeps updating. Without it, a cluster of misses at a small α (e.g. 99% VaR) drives
    α_t below 0, and the set stays infinite until α_t recovers at γ·α per step, which can take
    years. The floor keeps the sets finite at the cost of bound (4) during such clusters.
    """

    def __init__(
        self,
        ncm: NonConformity,
        alpha: float,
        gamma: float = 0.005,
        window: int | None = None,
        alpha_floor: float | None = None,
    ) -> None:
        super().__init__(ncm, alpha)
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        if alpha_floor is not None and not 0 < alpha_floor <= alpha:
            raise ValueError("alpha_floor must be in (0, alpha]")
        self.gamma, self.window, self.alpha_floor = gamma, window, alpha_floor

    def fit(self, f_cal, y_cal):
        self._buf: deque[float] = deque(self._scores(f_cal, y_cal), maxlen=self.window)
        self.alpha_t = self.alpha
        return self

    @property
    def q(self):
        scores = np.fromiter(self._buf, float)
        level = self.alpha_t
        if self.alpha_floor is not None:
            # At a level below 1/(n+1) even the largest score is not enough (q̂ = ∞), so the
            # effective floor is at least that: the set is then bounded by the largest score.
            level = max(level, self.alpha_floor, 1.0 / (len(scores) + 1))
        return conformal_quantile(scores, level)

    def update(self, f, y, covered):
        for c in np.atleast_1d(covered):
            self.alpha_t += self.gamma * (self.alpha - (0.0 if c else 1.0))  # eq. (3)
        self._buf.extend(self._scores(f, y))

    @property
    def scores(self):
        return np.fromiter(self._buf, float)


DTACI_GAMMAS = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128)


class DtACI(ConformalMethod):
    """Dynamically-tuned ACI (Gibbs & Candès, 2022): ACI without hand-picking γ.

    K ACI "experts" run in parallel, one per step size γ_k, each with its own level α_t^k. After
    each outcome the realized level β_t (the largest level whose set would still have covered
    y_t, ≈ share of calibration scores ≥ the new score) scores every expert with the pinball
    loss  ℓ(β, θ) = α(β − θ) − min(0, β − θ). Expert weights follow exponential weights with
    fixed-share mixing, and the working level is the weighted average α_t = Σ_k p_k α_t^k:

        p_k ← (1 − σ)·p_k e^{−η ℓ_k} / W + σ/K,    η = √(3/I) · √((log(KI) + 2) / ((1−α)² α³))

    with I the interval over which it should track a changing distribution and σ = 1/(2I).
    ``alpha_floor`` works as in :class:`AdaptiveConformal`.

    With delayed feedback (h-day targets), each expert is scored on the level it *issued* the
    forecast with, not its current level. Scoring on current levels under-covered with
    overlapping 5-day targets (93.5% at α = 5% in simulation, vs 94.6% with issued levels).
    ``stride`` > 1 updates experts on every stride-th outcome only; it did not help in testing
    and defaults to 1.
    """

    def __init__(
        self,
        ncm: NonConformity,
        alpha: float,
        gammas: tuple[float, ...] = DTACI_GAMMAS,
        interval: int = 500,
        window: int | None = None,
        alpha_floor: float | None = None,
        stride: int = 1,
    ) -> None:
        super().__init__(ncm, alpha)
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride
        self.gammas = np.asarray(gammas, dtype=float)
        k, i = len(self.gammas), interval
        self.eta = np.sqrt(3 / i) * np.sqrt((np.log(k * i) + 2) / ((1 - alpha) ** 2 * alpha**3))
        self.share = 1 / (2 * i)
        self.window, self.alpha_floor = window, alpha_floor

    def fit(self, f_cal, y_cal):
        self._buf: deque[float] = deque(self._scores(f_cal, y_cal), maxlen=self.window)
        self.experts = np.full(len(self.gammas), self.alpha)
        self.weights = np.full(len(self.gammas), 1 / len(self.gammas))
        self._seen = 0
        self._issued: deque[np.ndarray] = deque()  # expert levels at each prediction
        return self

    def predict(self, f):
        # Remember the expert levels in force when this forecast is issued: with delayed
        # feedback, each expert must be scored on the level it actually used, not its current one.
        for _ in range(len(f)):
            self._issued.append(self.experts.copy())
        return super().predict(f)

    @property
    def alpha_t(self) -> float:
        return float(self.weights @ self.experts)

    @property
    def scores(self):
        return np.fromiter(self._buf, float)

    @property
    def q(self):
        scores = self.scores
        level = self.alpha_t
        if self.alpha_floor is not None:
            level = max(level, self.alpha_floor, 1.0 / (len(scores) + 1))
        return conformal_quantile(scores, level)

    def update(self, f, y, covered):
        for s_new in self._scores(f, y):
            self._seen += 1
            if (self._seen - 1) % self.stride:
                if self._issued:
                    self._issued.popleft()
                self._buf.append(s_new)
                continue
            used = self._issued.popleft() if self._issued else self.experts
            buf = self.scores
            beta = np.sum(buf >= s_new) / (len(buf) + 1)
            diff = beta - used
            loss = self.alpha * diff - np.minimum(0.0, diff)
            w = self.weights * np.exp(-self.eta * (loss - loss.min()))  # shift for stability
            w = w / w.sum()
            self.weights = (1 - self.share) * w + self.share / len(w)
            miss = (used > beta).astype(float)
            self.experts = self.experts + self.gammas * (self.alpha - miss)
            self._buf.append(s_new)


class MondrianConformal(ConformalMethod):
    """Split conformal with one threshold per regime (a Mondrian conformal predictor).

    Calibration scores are split by ``Forecast.group`` (e.g. tercile of forecast volatility) and
    each group gets its own q̂_g. Under exchangeability *within* each group this gives coverage
    1 − α conditional on the group, not just on average: the fix for intervals that over-cover
    calm days and under-cover volatile ones.
    """

    def fit(self, f_cal, y_cal):
        if f_cal.group is None:
            raise ValueError("Mondrian conformal needs Forecast.group")
        s = self._scores(f_cal, y_cal)
        self._s = s
        self._qg = {g: conformal_quantile(s[f_cal.group == g], self.alpha)
                    for g in np.unique(f_cal.group)}  # fmt: skip
        self._q_all = conformal_quantile(s, self.alpha)
        return self

    @property
    def q(self):
        return self._q_all  # pooled threshold (used only by tail_mean)

    @property
    def scores(self):
        return self._s

    def predict(self, f):
        self.ncm.check(f)
        if f.group is None:
            raise ValueError("Mondrian conformal needs Forecast.group")
        q = np.array([self._qg.get(g, self._q_all) for g in f.group])
        return self.ncm.interval(f, q)


def regime_groups(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Group label 0..len(thresholds) for each value (e.g. terciles from calibration data)."""
    return np.digitize(np.asarray(values, float), thresholds).astype(float)


# --------------------------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------------------------


@dataclass
class OnlineResult:
    lo: np.ndarray
    hi: np.ndarray
    covered: np.ndarray
    alpha_path: np.ndarray | None  # α_t used at each step (ACI only)
    tail: np.ndarray | None = None  # conformal tail mean at each step (record_tail=True)


def run_online(
    method: ConformalMethod,
    f: Forecast,
    y: np.ndarray,
    feedback_lag: int = 1,
    record_tail: bool = False,
) -> OnlineResult:
    """Predict one step at a time, revealing outcome y_j only ``feedback_lag`` steps later.

    For an h-day-ahead target, y_j is known on day j + h, so use ``feedback_lag = h``. The
    method never sees y_j before predicting step j + feedback_lag.
    """
    if feedback_lag < 1:
        raise ValueError("feedback_lag must be >= 1")
    y = np.asarray(y, dtype=float)
    n = len(y)
    if len(f) != n:
        raise ValueError("forecast and outcome lengths differ")
    lo, hi = np.empty(n), np.empty(n)
    track_alpha = hasattr(method, "alpha_t") and isinstance(method, AdaptiveConformal | DtACI)
    alpha_path = np.empty(n) if track_alpha else None
    tail = np.empty(n) if record_tail else None

    for t in range(n):
        j = t - feedback_lag
        if j >= 0:
            yj = y[j : j + 1]
            method.update(f[j : j + 1], yj, is_covered(yj, lo[j : j + 1], hi[j : j + 1]))
        if track_alpha:
            alpha_path[t] = method.alpha_t
        lo_t, hi_t = method.predict(f[t : t + 1])
        lo[t], hi[t] = lo_t[0], hi_t[0]
        if record_tail:
            tail[t] = method.tail_mean()

    return OnlineResult(lo, hi, is_covered(y, lo, hi), alpha_path, tail)


_NCMS: dict[str, type[NonConformity]] = {
    "absolute": AbsoluteResidual,
    "normalized": NormalizedResidual,
    "cqr": QuantileResidual,
    "lower_tail": LowerTailResidual,
}


def build_method(
    name: str,
    alpha: float,
    *,
    rolling_window: int = 500,
    aci_gamma: float = 0.005,
    base_ncm: str = "absolute",
    aci_floor_ratio: float | None = None,
) -> ConformalMethod:
    """Map a config method name (``ConformalConfig.methods``) to a calibrator.

    ``base_ncm`` picks the score used by the online methods ``rolling`` and ``aci``.
    """
    match name:
        case "split_absolute":
            return SplitConformal(AbsoluteResidual(), alpha)
        case "split_normalized":
            return SplitConformal(NormalizedResidual(), alpha)
        case "cqr":
            return SplitConformal(QuantileResidual(), alpha)
        case "one_sided_var":
            return SplitConformal(LowerTailResidual(), alpha)
        case "rolling":
            return RollingConformal(_NCMS[base_ncm](), alpha, window=rolling_window)
        case "aci":
            floor = alpha * aci_floor_ratio if aci_floor_ratio else None
            return AdaptiveConformal(_NCMS[base_ncm](), alpha, gamma=aci_gamma, alpha_floor=floor)
        case "dtaci":
            floor = alpha * aci_floor_ratio if aci_floor_ratio else None
            return DtACI(_NCMS[base_ncm](), alpha, alpha_floor=floor)
        case "mondrian":
            return MondrianConformal(_NCMS[base_ncm](), alpha)
    raise ValueError(f"unknown conformal method {name!r}")
