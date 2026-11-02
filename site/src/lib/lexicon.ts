// The lexicon parser in the browser, ported from LexiconParser in src/aegis/slm_parser.py.
// Rules come from lexicon.json (written by `aegis export-site`); tests/lexicon.test.ts checks the
// port against Python output for a fixed set of headlines.

export interface LexiconRules {
  flags: string;
  direction: Record<string, [string, number][]>;
  positive: string;
  negative: string;
  risk_up: string;
  risk_down: string;
  strong: string;
  moderate: string;
  negation: string;
  sentences: string;
  exposure: Record<string, Record<string, number>>;
  aliases: Record<string, string>;
}

export interface Parsed {
  shocks: Record<string, number>;
  sentiment: number;
  volDirection: "up" | "down" | "neutral";
  impacts: Record<string, number>;
  evidence: string[];
  riskWords: string[]; // the words behind volDirection (not part of the Python output)
}

export function makeParser(rules: LexiconRules) {
  const re = (p: string, g = false) => new RegExp(p, rules.flags + (g ? "g" : ""));
  const direction = Object.entries(rules.direction).map(
    ([f, list]) => [f, list.map(([p, s]) => [re(p, true), s] as const)] as const,
  );
  const positive = re(rules.positive, true), negative = re(rules.negative, true);
  const riskUp = re(rules.risk_up, true), riskDown = re(rules.risk_down, true);
  const strong = re(rules.strong), moderate = re(rules.moderate), negation = re(rules.negation);
  const sentencesRe = re(rules.sentences);
  const count = (r: RegExp, t: string) => [...t.matchAll(r)].length;
  const intensity = (s: string) => (strong.test(s) ? 1.0 : moderate.test(s) ? 0.4 : 0.7);

  return function parse(text: string): Parsed {
    const sentences = text.split(sentencesRe).filter((s) => s && s.trim());
    const raw: Record<string, number> = Object.fromEntries(direction.map(([f]) => [f, 0]));
    const evidence: string[] = [];
    for (const sent of sentences) {
      let hit = false;
      for (const [factor, list] of direction) {
        for (const [pattern, sign] of list) {
          for (const m of sent.matchAll(pattern)) {
            const flip = negation.test(sent.slice(0, m.index)) ? -1 : 1;
            raw[factor] += sign * flip * intensity(sent);
            hit = true;
          }
        }
      }
      if (hit && evidence.length < 3) evidence.push(sent.trim().slice(0, 160));
    }
    const shocks = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, Math.tanh(v)]));
    const pos = count(positive, text), neg = count(negative, text);
    const sentiment = pos + neg ? ((pos - neg) / (pos + neg)) * Math.min(1, (pos + neg) / 3) : 0;
    const net = count(riskUp, text) - count(riskDown, text);
    const impacts: Record<string, number> = {};
    for (const [t, expo] of Object.entries(rules.exposure)) {
      const v = Object.entries(shocks).reduce((a, [k, s]) => a + (expo[k] ?? 0) * s, 0);
      impacts[t] = Math.max(-1, Math.min(1, Math.tanh(v)));
    }
    return {
      shocks,
      sentiment: Math.max(-1, Math.min(1, sentiment)),
      volDirection: net > 0 ? "up" : net < 0 ? "down" : "neutral",
      impacts,
      evidence,
      riskWords: [...text.matchAll(riskUp), ...text.matchAll(riskDown)].map((m) => m[0]),
    };
  };
}
