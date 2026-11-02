// The table of contents. `ready` chapters are written; the rest are listed as in preparation.
export interface Chapter {
  n: string;
  id: string;
  title: string;
  short: string; // the running head's label
  ready: boolean;
}

export const chapters: Chapter[] = [
  { n: "0", id: "cover", short: "Summary", title: "Summary", ready: true },
  { n: "1", id: "why", short: "Why", title: "Why this matters", ready: true },
  { n: "2", id: "portfolio", short: "Portfolio", title: "The portfolio", ready: true },
  { n: "3", id: "forecast", short: "Forecast", title: "A forecast", ready: true },
  { n: "4", id: "intervals", short: "Intervals", title: "Honest intervals", ready: true },
  { n: "5", id: "adapt", short: "Shifts", title: "When markets shift", ready: true },
  { n: "6", id: "tails", short: "Tails", title: "Tail risk", ready: true },
  { n: "7", id: "regulator", short: "Regulator", title: "The regulator's view", ready: true },
  { n: "8", id: "sizing", short: "Sizing", title: "Sizing positions", ready: true },
  { n: "9", id: "text", short: "Text", title: "Do words help?", ready: true },
  { n: "10", id: "risk-factors", short: "Risks", title: "Risk factors", ready: true },
  { n: "11", id: "checks", short: "Checks", title: "How it was checked", ready: true },
];

/** Previous and next for a chapter id; a next chapter not yet written points at the contents. */
export function neighbours(id: string) {
  const i = chapters.findIndex((c) => c.id === id);
  return { prev: chapters[i - 1], next: chapters[i + 1] };
}
