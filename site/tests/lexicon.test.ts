// The browser lexicon parser must give Python's output for the same headlines.
import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";
import { makeParser } from "../src/lib/lexicon.ts";

const read = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const parse = makeParser(read("../public/data/lexicon.json"));
const close = (a: number, b: number, what: string) => assert.ok(Math.abs(a - b) < 1e-9, `${what}: ${a} != ${b}`);

for (const g of read("./lexicon_golden.json")) {
  test(`lexicon: ${g.text.slice(0, 40)}`, () => {
    const r = parse(g.text);
    for (const [k, v] of Object.entries<number>(g.shocks)) close(r.shocks[k], v, k);
    for (const [k, v] of Object.entries<number>(g.impacts)) close(r.impacts[k], v, k);
    close(r.sentiment, g.sentiment, "sentiment");
    assert.equal(r.volDirection, g.vol);
    assert.deepEqual(r.evidence, g.evidence);
  });
}
