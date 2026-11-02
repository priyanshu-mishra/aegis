<script lang="ts">
  // Exhibit J: the lexicon parser, live. Type or pick a text; the same rules the Python pipeline
  // uses turn it into six signed macro shocks and an implied move for each holding.
  import { makeParser, type LexiconRules } from "../lib/lexicon";

  interface Sample {
    label: string;
    text: string;
  }
  interface Props {
    rules: LexiconRules;
    samples: Sample[];
  }
  let { rules, samples }: Props = $props();

  const parse = makeParser(rules);
  let text = $state(samples[0].text);
  const out = $derived(parse(text));

  const FACTOR_NAMES: Record<string, string> = {
    rates: "Rates rising",
    inflation: "Inflation rising",
    growth: "Economy strengthening",
    energy: "Energy prices rising",
    geopolitics: "Geopolitical tension",
    credit: "Credit stress",
  };
  const W = 220;
  const px = (v: number) => (W / 2) + (v * (W / 2 - 8));
  const fmt = (v: number) => (Math.abs(v) < 0.005 ? "0" : `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`);
</script>

<div class="demo">
  <fieldset class="samples">
    <legend>Try a text</legend>
    {#each samples as s}
      <button type="button" class:on={text === s.text} onclick={() => (text = s.text)}>{s.label}</button>
    {/each}
  </fieldset>

  <label class="input">
    <span class="visually-hidden">Text to parse</span>
    <textarea bind:value={text} rows="5" spellcheck="false"></textarea>
  </label>

  <div class="results">
    <table>
      <caption class="cap">Macro shocks read from the text, −1 to +1</caption>
      <tbody>
        {#each Object.entries(out.shocks) as [k, v]}
          <tr>
            <th scope="row">{FACTOR_NAMES[k] ?? k}</th>
            <td class="axis" aria-hidden="true">
              <svg viewBox={`0 0 ${W} 16`} width={W} height="16">
                <line x1="8" x2={W - 8} y1="8" y2="8" class="rule" />
                <line x1={W / 2} x2={W / 2} y1="2" y2="14" class="mid" />
                <circle cx={px(v)} cy="8" r="5" class={Math.abs(v) < 0.005 ? "dot zero" : "dot"} />
              </svg>
            </td>
            <td class="num">{fmt(v)}</td>
          </tr>
        {/each}
      </tbody>
    </table>

    <table>
      <caption class="cap">Implied move for each holding</caption>
      <tbody>
        {#each Object.entries(out.impacts) as [t, v]}
          <tr>
            <th scope="row">{t}</th>
            <td class="axis" aria-hidden="true">
              <svg viewBox={`0 0 ${W} 16`} width={W} height="16">
                <line x1="8" x2={W - 8} y1="8" y2="8" class="rule" />
                <line x1={W / 2} x2={W / 2} y1="2" y2="14" class="mid" />
                <circle cx={px(v)} cy="8" r="5" class={Math.abs(v) < 0.005 ? "dot zero" : "dot"} />
              </svg>
            </td>
            <td class="num">{fmt(v)}</td>
          </tr>
        {/each}
        <tr>
          <th scope="row">Volatility</th>
          <td colspan="2" class="vol">{out.volDirection === "up" ? "expected to rise" : out.volDirection === "down" ? "expected to fall" : "no signal"}</td>
        </tr>
      </tbody>
    </table>
  </div>

  {#if out.evidence.length}
    <p class="evidence"><span class="sc">Shock rules matched</span> {#each out.evidence as e, i}{i ? " · " : ""}“{e}”{/each}</p>
  {:else}
    <p class="evidence"><span class="sc">Shock rules matched</span> none, so every shock is zero.</p>
  {/if}
  <p class="evidence"><span class="sc">Volatility words</span> {out.riskWords.length ? out.riskWords.map((w) => `“${w}”`).join(", ") : "none"}</p>
</div>

<style>
  .samples {
    border: 0;
    padding: 0;
    margin: 0 0 var(--s0);
    display: flex;
    flex-wrap: wrap;
    gap: var(--s0);
  }
  legend {
    font-size: var(--step--1);
    color: var(--ink-3);
    margin-bottom: var(--s-1);
  }
  button {
    font: inherit;
    font-size: var(--step--1);
    background: var(--paper);
    color: var(--ink-2);
    border: 1px solid var(--rule);
    border-radius: 0;
    padding: 0.4rem 0.7rem;
    min-height: 2.5rem;
    cursor: pointer;
  }
  button:hover {
    border-color: var(--ink);
    color: var(--ink);
  }
  button.on {
    border-color: var(--ink);
    color: var(--ink);
    font-weight: 600;
  }
  textarea {
    width: 100%;
    font: inherit;
    font-size: 1rem;
    line-height: 1.5;
    color: var(--ink);
    background: var(--paper);
    border: 1px solid var(--rule-strong);
    border-radius: 0;
    padding: var(--s0);
    resize: vertical;
  }
  .results {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr));
    gap: var(--s1) var(--s3);
    margin-top: var(--s1);
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: var(--step--1);
  }
  .cap {
    caption-side: top;
    text-align: left;
    font-variant-caps: all-small-caps;
    letter-spacing: 0.08em;
    color: var(--ink-2);
    padding-bottom: var(--s-1);
    border-bottom: 1px solid var(--rule-strong);
  }
  th,
  td {
    padding: 0.2rem 0;
    border-bottom: 1px solid var(--rule);
    text-align: left;
    font-weight: 400;
  }
  th {
    color: var(--ink-2);
    white-space: nowrap;
    padding-right: var(--s0);
  }
  .axis {
    width: 100%;
  }
  .axis svg {
    display: block;
    width: 100%;
    max-width: 220px;
    height: 16px;
  }
  .num {
    text-align: right;
    white-space: nowrap;
    padding-left: var(--s0);
    color: var(--ink);
  }
  .vol {
    color: var(--ink);
  }
  .rule {
    stroke: var(--rule);
  }
  .mid {
    stroke: var(--ink-3);
  }
  .dot {
    fill: var(--ink);
  }
  .dot.zero {
    fill: var(--ghost);
  }
  /* On a phone the dot axis would push the figures off screen; the figures carry the reading. */
  @media (max-width: 480px) {
    .axis {
      display: none;
    }
  }
  .evidence {
    margin: var(--s0) 0 0;
    font-size: var(--step--1);
    color: var(--ink-2);
  }
  .evidence .sc {
    color: var(--ink-3);
  }
</style>
