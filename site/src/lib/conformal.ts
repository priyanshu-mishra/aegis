// Split conformal and ACI in the browser, ported line for line from src/aegis/conformal_ncm.py
// (absolute-residual score). tests/conformal.test.ts checks them against golden.json.

/** Conformal threshold q̂: the ⌈(n+1)(1−α)⌉-th smallest score of a sorted array. */
export function quantileSorted(sorted: number[], alpha: number): number {
  const n = sorted.length;
  if (n === 0) throw new Error("need at least one calibration score");
  if (alpha <= 0) return Infinity;
  if (alpha >= 1) return -Infinity;
  // The epsilon matches Python: float noise must not push k up by one.
  const k = Math.ceil((n + 1) * (1 - alpha) - 1e-9);
  return k > n ? Infinity : sorted[k - 1];
}

export const scores = (y: number[], mu: number[]) => y.map((v, i) => Math.abs(v - mu[i]));

export function splitQ(yCal: number[], muCal: number[], alpha: number): number {
  return quantileSorted(scores(yCal, muCal).sort((a, b) => a - b), alpha);
}

export function coverage(y: number[], lo: number[], hi: number[]): number {
  let c = 0;
  for (let i = 0; i < y.length; i++) if (lo[i] <= y[i] && y[i] <= hi[i]) c++;
  return c / y.length;
}

export interface AciRun {
  lo: number[];
  hi: number[];
  covered: boolean[];
  alphaPath: number[];
}

/** Insert v into an ascending array, keeping it sorted. */
function insertSorted(a: number[], v: number) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] < v) lo = m + 1;
    else hi = m;
  }
  a.splice(lo, 0, v);
}

/**
 * Adaptive Conformal Inference (Gibbs and Candès, 2021), run online over the test period.
 * Outcome j is revealed `lag` days later (an h-day target is known h days on).
 * α_{t+1} = α_t + γ(α − err_t); the quantile uses max(α_t, floor, 1/(n+1)).
 */
export function runAci(
  yCal: number[], muCal: number[], y: number[], mu: number[],
  alpha: number, gamma: number, lag: number, floorRatio: number | null,
): AciRun {
  const buf = scores(yCal, muCal).sort((a, b) => a - b);
  const floor = floorRatio ? alpha * floorRatio : null;
  let alphaT = alpha;
  const n = y.length;
  const lo = new Array<number>(n), hi = new Array<number>(n);
  const covered = new Array<boolean>(n), alphaPath = new Array<number>(n);
  for (let t = 0; t < n; t++) {
    const j = t - lag;
    if (j >= 0) {
      alphaT += gamma * (alpha - (covered[j] ? 0 : 1));
      insertSorted(buf, Math.abs(y[j] - mu[j]));
    }
    alphaPath[t] = alphaT;
    const level = floor == null ? alphaT : Math.max(alphaT, floor, 1 / (buf.length + 1));
    const q = quantileSorted(buf, level);
    lo[t] = mu[t] - q;
    hi[t] = mu[t] + q;
    covered[t] = lo[t] <= y[t] && y[t] <= hi[t];
  }
  return { lo, hi, covered, alphaPath };
}
