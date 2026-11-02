<script lang="ts">
  // Exhibit E: split conformal, live. Past errors on the calibration years set the cut-off q̂;
  // the same q̂ is then put around every forecast in the test period and checked.
  import { scaleLinear } from "d3-scale";
  import { coverage, quantileSorted, scores } from "../lib/conformal";
  import { alpha, LEVELS } from "../lib/state";
  import { unpack } from "../lib/pack";

  interface Props {
    data: string; // lib/pack.ts
    calYears: string;
  }
  let { data, calYears }: Props = $props();
  const { calY, calMu, dates, y, mu } = unpack(data);

  const sorted = scores(calY, calMu).sort((a, b) => a - b);
  const years = [...new Set(dates.map((d) => d.slice(0, 4)))];
  const yearIdx = years.map((yr) => dates.map((d, i) => (d.startsWith(yr) ? i : -1)).filter((i) => i >= 0));

  const level = $derived(Math.round((1 - $alpha) * 100));
  const setLevel = (v: number) => alpha.set((100 - v) / 100);

  const q = $derived(quantileSorted(sorted, $alpha));
  const held = $derived.by(() => {
    const lo = mu.map((m) => m - q), hi = mu.map((m) => m + q);
    const all = coverage(y, lo, hi);
    const byYear = yearIdx.map((idx) => idx.filter((i) => lo[i] <= y[i] && y[i] <= hi[i]).length / idx.length);
    return { all, byYear };
  });
  const target = $derived(1 - $alpha);

  // Histogram of calibration errors: bar length is a count of days.
  let w = $state(720);
  const narrow = $derived(w < 520);
  const H = 220;
  const m = { top: 18, right: 12, bottom: 30, left: 36 };
  const max = sorted[sorted.length - 1];
  const nb = 36;
  const bw = max / nb;
  const bins = Array.from({ length: nb }, (_, i) => sorted.filter((s) => s >= i * bw && (i === nb - 1 ? s <= max : s < (i + 1) * bw)).length);
  const x = $derived(scaleLinear().domain([0, max]).range([m.left, w - m.right]));
  const yS = scaleLinear().domain([0, Math.max(...bins)]).nice().range([H - m.bottom, m.top]);
  const outside = $derived(sorted.filter((s) => s > q).length);

  // Year strip: one row per test year, coverage against the promise.
  const rowH = 26;
  const YH = years.length * rowH + 34;
  const cx = $derived(scaleLinear().domain([0.7, 1]).range([narrow ? 44 : 60, w - 70]));
  const pc = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
</script>

<div class="pg" bind:clientWidth={w}>
  <label class="control">
    <span>Promise the range will hold on</span>
    <input
      type="range"
      min={LEVELS.min}
      max={LEVELS.max}
      step="1"
      value={level}
      oninput={(e) => setLevel(+e.currentTarget.value)}
      aria-valuetext={`${level}% of days`}
    />
    <output>{level}% of days</output>
  </label>

  <svg viewBox={`0 0 ${w} ${H}`} width={w} height={H} role="img"
    aria-label={`Histogram of ${sorted.length} forecast errors from ${calYears}. The cut-off is at ${q.toFixed(2)}; ${outside} errors lie beyond it.`}>
    {#each yS.ticks(4) as t}
      <line x1={m.left} x2={w - m.right} y1={yS(t)} y2={yS(t)} class="grid" />
      <text x={m.left - 6} y={yS(t)} class="tick" text-anchor="end" dominant-baseline="middle">{t}</text>
    {/each}
    {#each bins as c, i}
      {@const x0 = x(i * bw)}
      <rect x={x0 + 1} width={Math.max(0, x((i + 1) * bw) - x0 - 2)} y={yS(c)} height={yS(0) - yS(c)}
        class={(i + 0.5) * bw > q ? "bar out" : "bar"} />
    {/each}
    <line x1={x(q)} x2={x(q)} y1={m.top - 10} y2={H - m.bottom} class="cut" />
    <text x={x(q) + (x(q) > w * (narrow ? 0.5 : 0.7) ? -6 : 6)} y={m.top - 2} class="label" text-anchor={x(q) > w * (narrow ? 0.5 : 0.7) ? "end" : "start"}>
      {narrow ? `q̂ = ${q.toFixed(2)}, ${outside} days beyond` : `cut-off q̂ = ${q.toFixed(2)}, ${outside} of ${sorted.length} days beyond`}
    </text>
    {#each x.ticks(narrow ? 4 : 8) as t}
      <text x={x(t)} y={H - 10} class="tick" text-anchor="middle">{t}</text>
    {/each}
  </svg>
  <p class="axis-note">Size of each past miss (distance between forecast and outcome, in log units), {calYears}. Bar height: days.</p>

  <dl class="readout">
    <div><dt>Range around each forecast</dt><dd>÷{Math.exp(q).toFixed(2)} to ×{Math.exp(q).toFixed(2)}</dd></div>
    <div><dt>Promised</dt><dd>{pc(target, 0)}</dd></div>
    <div><dt>Held on test days, {years[0]}–{years.at(-1)}</dt><dd class:short={held.all < target - 0.02}>{pc(held.all)}</dd></div>
  </dl>

  <svg viewBox={`0 0 ${w} ${YH}`} width={w} height={YH} role="img"
    aria-label={`Share of days the range held in each test year: ${years.map((yr, i) => `${yr} ${pc(held.byYear[i])}`).join(", ")}.`}>
    {#each [0.7, 0.8, 0.9, 1] as t}
      <line x1={cx(t)} x2={cx(t)} y1="14" y2={YH - 24} class="grid" />
      <text x={cx(t)} y={YH - 6} class="tick" text-anchor="middle">{pc(t, 0)}</text>
    {/each}
    <line x1={cx(target)} x2={cx(target)} y1="6" y2={YH - 24} class="target" />
    <text x={cx(target) + 5} y="10" class="tick">promised</text>
    {#each years as yr, i}
      {@const cy = 24 + i * rowH}
      {@const v = held.byYear[i]}
      {@const miss = v < target - 0.02}
      <text x={cx.range()[0] - 10} y={cy} class="name" text-anchor="end" dominant-baseline="middle">{yr}</text>
      <circle cx={cx(Math.max(0.7, v))} {cy} r="5" class={miss ? "dot miss" : "dot"} />
      <text x={cx(Math.max(0.7, v)) + (v < target ? -10 : 10)} y={cy} class="val" text-anchor={v < target ? "end" : "start"} dominant-baseline="middle">{pc(v)}</text>
    {/each}
  </svg>
  <p class="axis-note">Share of days the range held, by year. Red: more than 2 points short of the promise.</p>
</div>

<style>
  .pg {
    width: 100%;
  }
  svg {
    display: block;
    overflow: visible;
    max-width: 100%;
    height: auto;
  }
  .control {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s0) var(--s1);
    margin-bottom: var(--s1);
    font-size: 1rem;
  }
  input[type="range"] {
    flex: 1 1 12rem;
    accent-color: var(--ink);
    min-height: 2.75rem;
  }
  output {
    font-weight: 600;
    min-width: 7.5em;
  }
  .grid {
    stroke: var(--rule);
  }
  .tick,
  .val,
  .label {
    font-size: 13px;
    fill: var(--ink-3);
    font-family: var(--serif);
  }
  .label {
    fill: var(--ink);
  }
  .name {
    font-size: 14px;
    fill: var(--ink-2);
    font-family: var(--serif);
  }
  .bar {
    fill: var(--ink-2);
  }
  .bar.out {
    fill: var(--ghost);
  }
  .cut,
  .target {
    stroke: var(--ink);
    stroke-dasharray: 4 3;
  }
  .dot {
    fill: var(--ink);
  }
  .dot.miss {
    fill: var(--paper);
    stroke: var(--herring);
    stroke-width: 2;
  }
  .axis-note {
    margin: var(--s-1) 0 var(--s2);
    font-size: var(--step--1);
    color: var(--ink-3);
  }
  .readout {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
    gap: var(--s1);
    margin: 0 0 var(--s2);
    padding: var(--s0) 0;
    border-top: 1px solid var(--rule-strong);
    border-bottom: 1px solid var(--rule);
  }
  .readout dt {
    font-size: var(--step--1);
    color: var(--ink-3);
  }
  .readout dd {
    margin: 0;
    font-size: var(--step-1);
    font-weight: 600;
  }
  .readout dd.short {
    color: var(--herring);
  }
</style>
