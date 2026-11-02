// Pure SVG path helpers for build-time exhibits (no DOM needed).
import { scaleLog, scaleTime, scaleLinear } from "d3-scale";
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
  return scaleTime()
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
    area<Pt>()
      .defined((p) => p.lo != null && p.hi != null)
      .x((p) => p.x)
      .y0((p) => p.lo as number)
      .y1((p) => p.hi as number)(pts) ?? ""
  );
}

export function linePath(pts: Pt[]): string {
  return (
    line<Pt>()
      .defined((p) => p.y != null)
      .x((p) => p.x)
      .y((p) => p.y as number)(pts) ?? ""
  );
}
