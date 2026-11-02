// The playgrounds' data, packed into one JSON string prop. Astro tags every array element when it
// serializes props, which doubled the page; a string goes through as is. Dates travel as day
// offsets from the first date.
import type { Frame } from "./data";

export interface PlaygroundData {
  calY: number[];
  calMu: number[];
  dates: string[];
  y: number[];
  mu: number[];
}

const DAY = 86_400_000;

export function pack(cal: Frame, vol: Frame): string {
  const t0 = Date.parse(vol.dates[0]);
  return JSON.stringify({
    start: vol.dates[0],
    d: vol.dates.map((d) => Math.round((Date.parse(d) - t0) / DAY)),
    y: vol.series.y,
    mu: vol.series.mu,
    calY: cal.series.y,
    calMu: cal.series.mu,
  });
}

export function unpack(s: string): PlaygroundData {
  const p = JSON.parse(s);
  const t0 = Date.parse(p.start);
  return {
    calY: p.calY,
    calMu: p.calMu,
    dates: p.d.map((k: number) => new Date(t0 + k * DAY).toISOString().slice(0, 10)),
    y: p.y,
    mu: p.mu,
  };
}
