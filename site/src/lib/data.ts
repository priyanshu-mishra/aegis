// Build-time access to the exported data (site/public/data, schema v1; see site/README.md).
import { readFileSync } from "node:fs";
import { join } from "node:path";

const DATA = join(process.cwd(), "public", "data");

export function load<T = any>(name: string): T {
  return JSON.parse(readFileSync(join(DATA, `${name}.json`), "utf8")) as T;
}

export interface Frame {
  dates: string[];
  series: Record<string, (number | null)[]>;
}

/** Rows of a columnar frame whose date falls in [from, to] (ISO strings, inclusive). */
export function slice(frame: Frame, from: string, to: string): Frame {
  const keep = frame.dates.map((d, i) => (d >= from && d <= to ? i : -1)).filter((i) => i >= 0);
  const series: Frame["series"] = {};
  for (const [k, v] of Object.entries(frame.series)) series[k] = keep.map((i) => v[i]);
  return { dates: keep.map((i) => frame.dates[i]), series };
}

/** log daily vol → annualized percent. */
export const annual = (x: number | null) => (x == null ? null : Math.exp(x) * Math.sqrt(252) * 100);

export const pct = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;

export function formatDate(iso: string): string {
  return new Date(iso + "T00:00:00Z").toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}
