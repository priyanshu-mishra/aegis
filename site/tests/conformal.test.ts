// The browser code must reproduce Python's reference values (golden.json) from the exported data.
import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";
import { coverage, runAci, splitQ } from "../src/lib/conformal.ts";

const load = (n: string) => JSON.parse(readFileSync(new URL(`../public/data/${n}.json`, import.meta.url), "utf8"));
const golden = load("golden");
const cal = load("calibration").series;
const vol = load("vol").series;
const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);

for (const [a, g] of Object.entries<any>(golden.levels)) {
  const alpha = Number(a);
  test(`split conformal at α=${a}`, () => {
    const q = splitQ(cal.y, cal.mu, alpha);
    close(q, g.split_q);
    const lo = vol.mu.map((m: number) => m - q), hi = vol.mu.map((m: number) => m + q);
    close(coverage(vol.y, lo, hi), g.split_coverage);
  });
  test(`ACI at α=${a}`, () => {
    const r = runAci(cal.y, cal.mu, vol.y, vol.mu, alpha, golden.gamma, golden.feedback_lag, golden.floor_ratio);
    close(r.covered.filter(Boolean).length / r.covered.length, g.aci_coverage);
    close(r.lo[0], g.aci_first[0]);
    close(r.hi[0], g.aci_first[1]);
    close(r.lo.at(-1)!, g.aci_last[0]);
    close(r.hi.at(-1)!, g.aci_last[1]);
    close(r.alphaPath.at(-1)!, g.aci_alpha_last);
  });
}
