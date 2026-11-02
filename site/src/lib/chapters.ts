// The table of contents. `ready` chapters are written; the rest are listed as in preparation.
export interface Chapter {
  n: string;
  id: string;
  title: string;
  ready: boolean;
}

export const chapters: Chapter[] = [
  { n: "0", id: "cover", title: "Summary", ready: true },
  { n: "1", id: "why", title: "Why this matters", ready: true },
  { n: "2", id: "portfolio", title: "The portfolio", ready: true },
  { n: "3", id: "forecast", title: "A forecast", ready: true },
  { n: "4", id: "intervals", title: "Honest intervals", ready: true },
  { n: "5", id: "adapt", title: "When markets shift", ready: true },
  { n: "6", id: "tails", title: "Tail risk", ready: false },
  { n: "7", id: "regulator", title: "The regulator's view", ready: false },
  { n: "8", id: "sizing", title: "Sizing positions", ready: false },
  { n: "9", id: "text", title: "Do words help?", ready: false },
  { n: "10", id: "risk-factors", title: "Risk factors", ready: false },
  { n: "11", id: "checks", title: "How it was checked", ready: false },
];

/** Previous and next for a chapter id; a next chapter not yet written points at the contents. */
export function neighbours(id: string) {
  const i = chapters.findIndex((c) => c.id === id);
  return { prev: chapters[i - 1], next: chapters[i + 1] };
}
