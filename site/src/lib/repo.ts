// Build-time facts read from the repository itself, so the site never restates them by hand.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const TESTS = join(process.cwd(), "..", "tests");

/** Test functions in the Python suite, and those that guard against look-ahead. */
export function testCounts() {
  let all = 0, leakage = 0;
  for (const f of readdirSync(TESTS).filter((f) => f.startsWith("test_") && f.endsWith(".py"))) {
    for (const m of readFileSync(join(TESTS, f), "utf8").matchAll(/^def (test_\w+)/gm)) {
      all++;
      if (/lookahead|future|as_of|leak/.test(m[1]) || f === "test_no_lookahead.py") leakage++;
    }
  }
  return { all, leakage };
}
