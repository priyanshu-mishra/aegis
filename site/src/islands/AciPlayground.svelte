<script lang="ts">
  // Exhibit F: ACI, live. After each outcome the working level α_t moves by γ: a miss widens the
  // next ranges, a hit narrows them. Re-runs the whole test period in the browser.
  import { scaleLinear, scaleLog, scaleUtc } from "d3-scale";
  import { area, line } from "d3-shape";
  import { quantileSorted, runAci, scores } from "../lib/conformal";
  import { alpha, LEVELS } from "../lib/state";
  import { unpack } from "../lib/pack";

  interface Props {
    data: string; // lib/pack.ts
    lag: number;
    floorRatio: number | null;
    defaultGamma: number;
    focusYear: string;
  }
  let { data, lag, floorRatio, defaultGamma, focusYear }: Props = $props();
  const { calY, calMu, dates, y, mu } = unpack(data);

  const GAMMAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1];
  let gi = $state(Math.max(0, GAMMAS.indexOf(defaultGamma)));
  const gamma = $derived(GAMMAS[gi]);
  const level = $derived(Math.round((1 - $alpha) * 100));

  const sorted = scores(calY, calMu).sort((a, b) => a - b);
  const run = $derived(runAci(calY, calMu, y, mu, $alpha, gamma, lag, floorRatio));
  const qSplit = $derived(quantileSorted(sorted, $alpha));

  const fIdx = dates.map((d, i) => (d.startsWith(focusYear) ? i : -1)).filter((i) => i >= 0);
  const held = (idx: number[], lo: (i: number) => number, hi: (i: number) => number) =>
    idx.filter((i) => lo(i) <= y[i] && y[i] <= hi(i)).length / idx.length;
  const all = dates.map((_, i) => i);
  const stats = $derived({
    aciAll: run.covered.filter(Boolean).length / run.covered.length,
    aciYear: held(fIdx, (i) => run.lo[i], (i) => run.hi[i]),
    splitAll: held(all, (i) => mu[i] - qSplit, (i) => mu[i] + qSplit),
    splitYear: held(fIdx, (i) => mu[i] - qSplit, (i) => mu[i] + qSplit),
  });

  let w = $state(720);
  const narrow = $derived(w < 520);
  const ann = (v: number) => Math.exp(v) * Math.sqrt(252) * 100;
  const pc = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

  // Chart 1: the focus year, ACI range against the fixed textbook range.
  const H1 = $derived(narrow ? 260 : 330);
  const m = { top: 12, right: 12, bottom: 28, left: 40 };
  const x1 = $derived(scaleUtc().domain([new Date(dates[fIdx[0]]), new Date(dates[fIdx.at(-1)!])]).range([m.left, w - m.right]));
  const y1 = $derived(scaleLog().domain([2.5, 160]).range([H1 - m.bottom, m.top]).clamp(true));
  const pts = $derived(fIdx.map((i) => ({
    x: x1(new Date(dates[i])),
    lo: y1(ann(run.lo[i])),
    hi: y1(ann(run.hi[i])),
    sLo: y1(ann(mu[i] - qSplit)),
    sHi: y1(ann(mu[i] + qSplit)),
    r: y1(ann(y[i])),
    miss: !run.covered[i],
  })));
  type P = (typeof pts)[number];
  const band = $derived(area<P>().digits(1).x((p) => p.x).y0((p) => p.lo).y1((p) => p.hi)(pts) ?? "");
  const real = $derived(line<P>().digits(1).x((p) => p.x).y((p) => p.r)(pts) ?? "");
  const sUp = $derived(line<P>().digits(1).x((p) => p.x).y((p) => p.sHi)(pts) ?? "");
  const sDn = $derived(line<P>().digits(1).x((p) => p.x).y((p) => p.sLo)(pts) ?? "");
  const fmtM = (d: Date) => d.toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" });

  // Chart 2: the working level α_t over the whole test period.
  const H2 = 150;
  const x2 = $derived(scaleUtc().domain([new Date(dates[0]), new Date(dates.at(-1)!)]).range([m.left, w - m.right]));
  // Years labelled at mid-year, so the first, partial year (the crash) gets its label.
  const yFirst = +dates[0].slice(0, 4), yLast = +dates.at(-1)!.slice(0, 4);
  const years = $derived(
    Array.from({ length: yLast - yFirst + 1 }, (_, i) => yFirst + i)
      .filter((_, i) => !narrow || i % 2 === 0)
      .map((yr) => ({ yr, x: x2(new Date(Date.UTC(yr, 6, 1))) }))
      .filter((t) => t.x >= m.left && t.x <= w - m.right),
  );
  const aMax = $derived(Math.max($alpha * 2.2, ...run.alphaPath));
  const aMin = $derived(Math.min(0, ...run.alphaPath));
  const y2 = $derived(scaleLinear().domain([aMin, aMax]).nice().range([H2 - m.bottom, m.top]));
  const aPath = $derived(
    line<number>().digits(1).x((_, i) => x2(new Date(dates[i]))).y((a) => y2(a))(run.alphaPath) ?? "",
  );
</script>

<div class="pg" bind:clientWidth={w}>
  <div class="controls">
    <label class="control">
      <span>Promise</span>
      <input type="range" min={LEVELS.min} max={LEVELS.max} step="1" value={level}
        oninput={(e) => alpha.set((100 - +e.currentTarget.value) / 100)} aria-valuetext={`${level}% of days`} />
      <output>{level}% of days</output>
    </label>
    <label class="control">
      <span>Step size γ</span>
      <input type="range" min="0" max={GAMMAS.length - 1} step="1" bind:value={gi} aria-valuetext={`gamma ${gamma}`} />
      <output>{gamma}{gamma === defaultGamma ? ", the study's setting" : ""}</output>
    </label>
  </div>

  <svg viewBox={`0 0 ${w} ${H1}`} width={w} height={H1} role="img"
    aria-label={`${focusYear}: realized volatility against the adaptive range (held on ${pc(stats.aciYear)} of days) and the fixed range (${pc(stats.splitYear)}).`}>
    {#each [5, 10, 20, 40, 80, 160] as t}
      <line x1={m.left} x2={w - m.right} y1={y1(t)} y2={y1(t)} class="grid" />
      <text x={m.left - 6} y={y1(t)} class="tick" text-anchor="end" dominant-baseline="middle">{t}%</text>
    {/each}
    {#each x1.ticks(narrow ? 4 : 6) as t}
      <text x={x1(t)} y={H1 - 8} class="tick" text-anchor="middle">{fmtM(t)}</text>
    {/each}
    <path d={band} class="band" />
    <path d={sUp} class="ghost" />
    <path d={sDn} class="ghost" />
    <path d={real} class="realized" />
    {#each pts.filter((p) => p.miss) as p}
      <circle cx={p.x} cy={p.r} r="3.4" class="miss" />
    {/each}
  </svg>
  <p class="axis-note">{focusYear}, realized 5-day volatility (line, annualized, log scale) against the adaptive range (shaded) and the fixed textbook range (dashed). Circles: days outside the shaded range.</p>

  <dl class="readout">
    <div><dt>Adaptive, {focusYear}</dt><dd class:short={stats.aciYear < 1 - $alpha - 0.02}>{pc(stats.aciYear)}</dd></div>
    <div><dt>Fixed, {focusYear}</dt><dd class:short={stats.splitYear < 1 - $alpha - 0.02}>{pc(stats.splitYear)}</dd></div>
    <div><dt>Adaptive, all years</dt><dd class:short={stats.aciAll < 1 - $alpha - 0.02}>{pc(stats.aciAll)}</dd></div>
    <div><dt>Fixed, all years</dt><dd class:short={stats.splitAll < 1 - $alpha - 0.02}>{pc(stats.splitAll)}</dd></div>
  </dl>

  <svg viewBox={`0 0 ${w} ${H2}`} width={w} height={H2} role="img"
    aria-label={`The working level over ${dates[0].slice(0, 4)}–${dates.at(-1)!.slice(0, 4)}; it ends at ${pc(run.alphaPath.at(-1)!, 2)}.`}>
    {#each y2.ticks(3) as t}
      <line x1={m.left} x2={w - m.right} y1={y2(t)} y2={y2(t)} class="grid" />
      <text x={m.left - 6} y={y2(t)} class="tick" text-anchor="end" dominant-baseline="middle">{pc(t, 0)}</text>
    {/each}
    <line x1={m.left} x2={w - m.right} y1={y2($alpha)} y2={y2($alpha)} class="target" />
    <path d={aPath} class="realized" />
    {#each years as t}
      <text x={t.x} y={H2 - 8} class="tick" text-anchor="middle">{t.yr}</text>
    {/each}
  </svg>
  <p class="axis-note">The working level α<sub>t</sub>, {dates[0].slice(0, 4)}–{dates.at(-1)!.slice(0, 4)}. Dashed: the promised {pc($alpha, 0)}. When it dips, the model has been missing and widens its ranges.</p>
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
  .controls {
    display: grid;
    gap: var(--s0);
    margin-bottom: var(--s1);
  }
  .control {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s0) var(--s1);
    font-size: 1rem;
  }
  .control span {
    min-width: 6.5em;
  }
  input[type="range"] {
    flex: 1 1 12rem;
    accent-color: var(--ink);
    min-height: 2.75rem;
  }
  output {
    font-weight: 600;
    min-width: 9em;
  }
  .grid {
    stroke: var(--rule);
  }
  .tick {
    font-size: 13px;
    fill: var(--ink-3);
    font-family: var(--serif);
  }
  .band {
    fill: var(--band);
  }
  .ghost {
    fill: none;
    stroke: var(--ink-3);
    stroke-dasharray: 3 3;
  }
  .target {
    stroke: var(--ink);
    stroke-dasharray: 4 3;
  }
  .realized {
    fill: none;
    stroke: var(--ink);
    stroke-width: 1.5;
    stroke-linejoin: round;
  }
  .miss {
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
    grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr));
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
