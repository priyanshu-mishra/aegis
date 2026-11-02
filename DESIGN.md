---
name: Aegis
description: A walk-forward study of honest error bars for portfolio risk, set as a preliminary prospectus.
colors:
  paper: "#ffffff"
  ink: "#111111"
  ink-secondary: "#3d3d3d"
  ink-caption: "#6a6a6a"
  rule-hairline: "#d4d4d4"
  rule-strong: "#111111"
  herring-red: "#c0261d"
  band-tint: "rgba(17, 17, 17, 0.12)"
  ghost-grey: "#b5b5b5"
  wash: "#ebebe8"
typography:
  display:
    fontFamily: "Source Serif 4 Variable, Source Serif 4, Georgia, serif"
    fontSize: "clamp(3rem, 2rem + 4vw, 5.5rem)"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "0.32em"
    fontVariation: "\"opsz\" 60"
  headline:
    fontFamily: "Source Serif 4 Variable, Source Serif 4, Georgia, serif"
    fontSize: "2.25rem"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Source Serif 4 Variable, Source Serif 4, Georgia, serif"
    fontSize: "1.75rem"
    fontWeight: 400
    lineHeight: 1.2
  body:
    fontFamily: "Source Serif 4 Variable, Source Serif 4, Georgia, serif"
    fontSize: "1.125rem"
    fontWeight: 400
    lineHeight: 1.6
    fontFeature: "\"lnum\" 1, \"tnum\" 1"
  body-small:
    fontFamily: "Source Serif 4 Variable, Source Serif 4, Georgia, serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.4
  caption:
    fontFamily: "Source Serif 4 Variable, Source Serif 4, Georgia, serif"
    fontSize: "1.0625rem"
    fontWeight: 600
    lineHeight: 1.35
  label:
    fontFamily: "Source Serif 4 Variable, Source Serif 4, Georgia, serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0.08em"
    fontFeature: "\"c2sc\" 1, \"smcp\" 1"
  code:
    fontFamily: "ui-monospace, SF Mono, Menlo, Consolas, monospace"
    fontSize: "0.875em"
rounded:
  none: "0px"
spacing:
  s-1: "4px"
  s0: "8px"
  s1: "16px"
  s2: "24px"
  s3: "40px"
  s4: "64px"
  s5: "96px"
components:
  running-head:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink-secondary}"
    typography: "{typography.label}"
    height: "48px"
    rounded: "{rounded.none}"
  red-herring-legend:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.herring-red}"
    typography: "{typography.label}"
    width: "52px"
  exhibit-caption:
    textColor: "{colors.ink}"
    typography: "{typography.caption}"
  exhibit-source:
    textColor: "{colors.ink-caption}"
    typography: "{typography.label}"
    padding: "8px 0 0"
  chapter-opener:
    textColor: "{colors.ink}"
    typography: "{typography.headline}"
    padding: "24px 0 0"
  findings-table:
    textColor: "{colors.ink}"
    typography: "{typography.body-small}"
    padding: "8px 0"
  practitioner-notes:
    textColor: "{colors.ink-secondary}"
    typography: "{typography.body-small}"
    padding: "8px 0"
---

# Design System: Aegis

## Overview

**Creative North Star: "The Prospectus"**

Aegis is typeset as a preliminary prospectus for a model: white paper, black ink, one serif family, and a single red reserved for the warning that the figures are preliminary and for the places the model failed. Structure comes from the document trade, not the web: double rules open sections, hairlines divide table rows, labels are set in small caps, every chart is a lettered Exhibit with a caption, a Source line and a table view, and a running head carries the chapter list.

Density is that of a reading document. One text column of 66 characters carries the argument; exhibits may run wider than the column but never break out into decorated panels. Nothing floats, glows or rounds off. Depth is made by rule weight alone.

The world rejects the scrolling ML-explainer article: no hero figure, no cards, no shadows, no gradients, no rounded corners, no labels stacked above headings.

**Key Characteristics:**
- White paper (#ffffff), near-black ink (#111111), greys for hierarchy, one red.
- Source Serif 4 with optical sizes for everything; lining tabular figures throughout.
- A ladder of rule weights (double, heavy, ink hairline, grey hairline) replaces boxes and shadows.
- Lettered Exhibits, each with caption, Source line and "View as a table".
- Charts drawn twice: a wide drawing and a narrow one, swapped at 600px.

## Colors

A monochrome ink-on-paper palette with one reserved red.

### Primary
- **Herring Red** (#c0261d): the red-herring legend down the left margin, the miss circles on interval exhibits, and the dots of models whose breach rate is rejected. It marks a warning or a failure and nothing else; 5.9:1 on paper.

### Neutral
- **Paper** (#ffffff): the page, the running head, the legend column, and the fill of miss circles.
- **Ink** (#111111): body text, headings, the realized-volatility line, the focus method's dot, heavy rules.
- **Secondary Ink** (#3d3d3d): ledes, chapter questions, practitioner notes, running-head links, risk-factor text.
- **Caption Ink** (#6a6a6a): Source lines, chart ticks, chapter numbers, exhibit letters, target columns, link underlines at rest; 5.4:1 on paper.
- **Hairline** (#d4d4d4): table row rules, chart gridlines, the rule above a Source line, breach-chart stems.
- **Strong Rule** (#111111): double rules, exhibit top rules, table header rules, the running-head base rule.
- **Band Tint** (ink at 12%): the conformal interval band. The uncertainty range is ink, thinned, never a hue.
- **Ghost Grey** (#b5b5b5): methods not in focus. They recede, they never vanish.

### Named Rules
**The Red Herring Rule.** Red appears only on the preliminary legend, on misses, and on breaches or rejected models. If a red mark does not signal a warning or a failure, it is wrong.

**The Ink Band Rule.** Intervals and ranges are drawn as tinted ink; comparison methods are dashed Caption Ink edges. Data color comes from ink weight, not from a palette.

## Typography

**Display Font:** Source Serif 4 Variable (with Georgia, serif)
**Body Font:** Source Serif 4 Variable (with Georgia, serif)
**Label/Mono Font:** ui-monospace stack, inside practitioner code only

**Character:** One serif family across every role, with optical sizing on so the 88px cover caps and the 13px Source lines are each drawn for their size. Hierarchy comes from size, weight (400 and 600 only), italic and small caps, never from a second family.

### Hierarchy
- **Display** (600, clamp 48–88px, line-height 1, tracked 0.32em, uppercase, opsz 60): the AEGIS mark on the cover, once.
- **Headline** (600, 36px, 1.15, -0.01em): chapter titles, with the chapter number in 400 Caption Ink inline before the title.
- **Title** (400 italic, 28px, 1.2): the cover subtitle; set upright 600 at the same size for the contents heading.
- **Body** (400, 18px, 1.6): running prose in a 66ch measure. Chapter questions sit at 22px italic in Secondary Ink.
- **Body Small** (400, 16px, 1.4): findings table, risk-factors note, practitioner notes, pager.
- **Caption** (600, 17px, 1.35): exhibit titles, preceded inline by the exhibit letter in small caps.
- **Label** (400, 13px, all small caps, 0.08em): Source lines, table headers, Definitions, pager labels, the red-herring legend (0.1em), the running-head mark (0.24em, 600).

### Named Rules
**The One Family Rule.** Every word is Source Serif 4; the monospace stack appears only around code inside practitioner notes.

**The Tabular Figures Rule.** Lining tabular figures are on globally, so every number in prose, tables and charts aligns in columns.

**The Inline Label Rule.** Small-caps labels sit inline with or beside what they name ("Exhibit A · title", "Source …", "Risk factors."), never as a separate line stacked above a heading.

## Layout

The page reserves a 52px left margin for the red-herring legend; a sticky 48px running head sits across the top. The cover is a two-column grid (a 34rem offering-summary column and Exhibit A filling the rest, max 82rem), with the exhibit sticky beside the summary. Chapters are a single 50rem column; prose holds to a 66ch measure, while exhibits may span the full chapter width.

Spacing follows one 8px-based rhythm (4, 8, 16, 24, 40, 64, 96px): 8px inside captions and table cells, 24px between cover blocks, 40px around exhibits and notes, 64px between chapters.

At 960px and below, the legend rotates from the vertical margin into a horizontal band under a red hairline, the cover collapses to one column with Exhibit A below the summary, and two-column lists (Definitions, contents) become one. At 600px and below, every chart swaps to its narrow drawing.

**The Drawn Twice Rule.** Every chart ships two drawings from the same data: a wide 720-unit viewBox and a narrow 400-unit one with larger relative text and labels moved above rows. They toggle at 600px; a chart is never just scaled down.

## Elevation & Depth

The system is flat. There are no shadows, no tonal surfaces and no stacked panels; the page is one sheet of paper. Separation comes from rule weight and whitespace. The only overlap is the sticky running head and the fixed legend, both opaque paper with an ink or red rule edge. Chart labels use a paper-colored stroke halo so they read over gridlines; it is a knockout, not a shadow.

**The Flat Paper Rule.** Nothing lifts off the page. If an element needs emphasis, give it a heavier rule or more space, not elevation.

## Shapes

Corners are square everywhere (0px). Form language is entirely linear, a ladder of rules:
- **3px double ink rule:** opens a chapter, closes the cover lede, heads the contents.
- **2px solid ink rule:** the top edge of every Exhibit.
- **1px ink rule:** running-head base, table header underline, practitioner-notes top and bottom, byline and pager tops.
- **1px hairline (#d4d4d4):** table rows, the rule above a Source line, contents list rows, chart gridlines.
- **1.5px red rule:** the inner edge of the red-herring legend only.

Circles appear only as data marks in charts (miss rings, breach dots).

**The Rule Ladder Rule.** Each rule weight has one meaning; pick the weight by the boundary it marks, never for decoration.

## Components

### Running Head (Navigation)
A prospectus running head: sticky, 48px, paper with a 1px ink base rule. Left, the AEGIS mark in tracked small caps; centre, the chapter list in 13px Secondary Ink with Caption Ink numbers, unbuilt chapters collapsed into one "in preparation" entry; right, a Code link. Hover darkens to Ink and underlines. On narrow screens the list scrolls horizontally with no scrollbar.

### Red-Herring Legend
Fixed down the left margin in vertical small caps, red on paper, with a 1.5px red inner rule. It states the figures are preliminary, the data-through date, and whether the latest refresh failed. Below 960px it becomes a horizontal band above the running head.

### Exhibit (Signature)
- **Caption:** one line, "Exhibit A · title"; letter and separator in small-caps Caption Ink, title in 600 Ink.
- **Frame:** 2px ink top rule, no border, no background.
- **Source line:** 13px Caption Ink under a hairline, led by "Source" in small caps; describes encoding as well as provenance.
- **Table view:** a "View as a table" disclosure in Secondary Ink holding the full data table (right-aligned numbers, first column left, ink header rule, hairline rows).

**The Lettered Exhibit Rule.** Every chart is a lettered Exhibit with caption, Source line and table view, and carries an accessible label summarising what it shows.

### Chapter Opener
Double ink rule, then the numbered chapter title (headline), then the chapter's question in 22px Secondary Ink italic with the reading time in 13px Caption Ink trailing it.

### Findings Table
Result against target. Small-caps header over a 1px ink rule; rows on hairlines; the finding in 16px with its comparison beneath in 13px Caption Ink; the result right-aligned in 600, the target right-aligned in Caption Ink.

### Practitioner Notes
A disclosure between two 1px ink rules, its summary in small caps; the body is 16px Secondary Ink. Depth stays one click away and the plain-language layer reads without it.

### Links and Defined Terms
Links inherit text color with a 1px Caption Ink underline at 0.18em offset that darkens to Ink on hover. Defined terms use a dotted underline and link to the Definitions list. Keyboard focus is a 2px ink outline at 3px offset.

### Pager
Previous and next under a 1px ink rule, each link with its small-caps direction label on its own line; no underline until hover.

## Do's and Don'ts

### Do:
- **Do** reserve Herring Red for the legend, misses and breaches or rejected models (The Red Herring Rule).
- **Do** open every chapter with a 3px double ink rule and every exhibit with a 2px ink rule.
- **Do** letter every chart as an Exhibit with a one-line caption, a Source line and a "View as a table" disclosure.
- **Do** draw every chart twice, wide (720) and narrow (400), toggled at 600px.
- **Do** keep prose to a 66ch measure and figures tabular.
- **Do** draw the method in focus in Ink and the others in Ghost Grey.

### Don't:
- **Don't** use cards, panels or background fills to group content; use rules and space.
- **Don't** use shadows, gradients or rounded corners.
- **Don't** add a second typeface outside code.
- **Don't** set a label, kicker or eyebrow on its own line above a heading.
- **Don't** use red for emphasis, links, hover, selection or highlight states.
- **Don't** scale a wide chart down for phones; ship its narrow drawing.
