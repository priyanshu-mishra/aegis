// Median Lighthouse score per category over several runs, checked against a floor.
// Usage: node scripts/lighthouse_check.mjs lh1.json lh2.json lh3.json
// Accessibility, best practices and SEO are deterministic, so they must be perfect; performance
// varies between CI machines, so its floor leaves room for noise.
import { readFileSync } from "node:fs";

const FLOORS = { performance: 0.9, accessibility: 1, "best-practices": 1, seo: 1 };
const runs = process.argv.slice(2).map((p) => JSON.parse(readFileSync(p, "utf8")));
const median = (xs) => xs.sort((a, b) => a - b)[Math.floor(xs.length / 2)];

let failed = false;
console.log("### Lighthouse (mobile, median of " + runs.length + " runs)\n");
console.log("| Category | Score | Floor |\n|---|---|---|");
for (const [cat, floor] of Object.entries(FLOORS)) {
  const score = median(runs.map((r) => r.categories[cat].score));
  const ok = score >= floor;
  failed ||= !ok;
  console.log(`| ${cat} | ${Math.round(score * 100)}${ok ? "" : " ✗"} | ${floor * 100} |`);
}
const lcp = median(runs.map((r) => r.audits["largest-contentful-paint"].numericValue));
console.log(`\nLargest contentful paint: ${(lcp / 1000).toFixed(1)} s`);
process.exit(failed ? 1 : 0);
