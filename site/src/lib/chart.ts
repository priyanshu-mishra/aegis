// Pure SVG path helpers for build-time exhibits (no DOM needed).
import { scaleLog, scaleUtc, scaleLinear } from "d3-scale";
import { area, line } from "d3-shape";

export interface Box {
  width: number;
  height: number;
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export const box = (width: number, height: number, m: Partial<Box> = {}): Box => ({
  width,
  height,
  top: 12,
  right: 12,
  bottom: 28,
  left: 44,
  ...m,
});

export function timeX(dates: string[], b: Box) {
  return scaleUtc()
    .domain([new Date(dates[0]), new Date(dates[dates.length - 1])])
    .range([b.left, b.width - b.right]);
}

export function logY(lo: number, hi: number, b: Box) {
  return scaleLog().domain([lo, hi]).range([b.height - b.bottom, b.top]).clamp(true);
}

export function linearX(lo: number, hi: number, b: Box) {
  return scaleLinear().domain([lo, hi]).range([b.left, b.width - b.right]);
}

type Pt = { x: number; lo: number | null; hi: number | null; y: number | null };

export function bandPath(pts: Pt[]): string {
  return (
    area<Pt>().digits(1)
      .defined((p) => p.lo != null && p.hi != null)
      .x((p) => p.x)
      .y0((p) => p.lo as number)
      .y1((p) => p.hi as number)(pts) ?? ""
  );
}

export function linePath(pts: Pt[]): string {
  return (
    line<Pt>().digits(1)
      .defined((p) => p.y != null)
      .x((p) => p.x)
      .y((p) => p.y as number)(pts) ?? ""
  );
}

/**
 * A line through screen points, thinned to one pixel column at a time: each column keeps its
 * lowest and highest point, so spikes and troughs survive. Daily series with more
 * points than pixels shrink several-fold with no visible change. Null y breaks the line.
 */
export function thinPath(pts: { x: number; y: number | null }[]): string {
  const out: string[] = [];
  let col: { x: number; y: number; i: number }[] = [];
  let colX = NaN;
  let pen = false;
  let px = 0, py = 0; // last point written: later points are relative, which is shorter
  const f = (v: number) => Math.round(v); // whole pixels: at one point per column, finer is invisible
  const flush = () => {
    if (!col.length) return;
    const lo = col.reduce((a, b) => (b.y < a.y ? b : a));
    const hi = col.reduce((a, b) => (b.y > a.y ? b : a));
    const keep = [...new Set([lo, hi])].sort((a, b) => a.i - b.i);
    for (const p of keep) {
      const x = f(p.x), y = f(p.y);
      if (!pen) out.push(`M${x},${y}`);
      else if (x !== px || y !== py) out.push(`l${x - px},${y - py}`);
      px = x;
      py = y;
      pen = true;
    }
    col = [];
  };
  pts.forEach((p, i) => {
    if (p.y == null || !Number.isFinite(p.y)) {
      flush();
      pen = false;
      colX = NaN;
      return;
    }
    const c = Math.round(p.x);
    if (c !== colX) {
      flush();
      colX = c;
    }
    col.push({ x: p.x, y: p.y, i });
  });
  flush();
  return out.join("");
}
