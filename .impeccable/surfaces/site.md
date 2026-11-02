---
version: 1
slug: "site"
primary_target: "site"
related_targets: []
---

# Surface brief: Aegis public site

## Scope and mode

The public, guided site for Aegis (`site/`), one long page of numbered chapters plus anchors. Visitor mode: **Read**. The reader's question: "What did this person build, does it work, and why should I care?" Source material: `site/public/data/*.json` (exported nightly) and the findings in `README.md`. Wayfinding: a running head with the chapter list, numbered chapter openers, lettered exhibits, previous and next at each chapter end.

Audience and job (from PRODUCT.md): hiring teams for quant, risk and portfolio roles, giving it a few minutes; experts open the practitioner notes. First release of this surface (V5): the design system and chapters 0 (cover) and 1 (why this matters).

## Direction contract

THESIS: Aegis is set as a preliminary prospectus for a model. The cover states the findings like an offering summary, a red-herring legend says the figures are preliminary and refresh nightly, failures are disclosed under Risk Factors, and every chart is a lettered Exhibit. It refuses the category default: the scrolling ML-explainer article with a hero figure and sticky sans-serif contents.

OWN-WORLD: White paper and black ink. One red (#c0261d) used only for the red-herring legend, misses and breaches. A single serif family with optical sizes (display caps letterspaced on the cover, text size in the column), tabular figures everywhere. Double rules under section heads, hairline table rules, small-caps labels. No cards, shadows, gradients or rounded corners.

STORY: The reader sees the findings and the 2020 exhibit on the cover, understands in plain language why honest error bars matter, then follows numbered chapters through method, results, disclosure of failures, and how honesty was checked. They leave believing the work is rigorous and candid, and they can open the repo or LinkedIn.

FIRST VIEWPORT: Red vertical legend down the left margin. Centered cover column: AEGIS in wide-spaced display caps, the title in italic, then "A walk-forward study, 2020–2026" and a one-line description beneath it (no label above any heading), double rule, the findings table (result vs target), a short Risk Factors note linking to what didn't work. To the right, Exhibit A: the 2020 interval chart with its caption. Byline at the foot of the cover column. The way forward is the running head with the chapter list.

FORM: Prospectus (securities offering document), position 3 on my ordered list, seed: 3a19fb0dc34c6870.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

Raises kept from the declined worlds (as named in the decision round):
- Calendar Pad: the test period also reads month by month, each month with its own coverage.
- Dance Score: length is duration; a bar's length always encodes days.
- Daylight Section: every animated exhibit has a stated still frame (reduced motion, print).
- Miura Sheet: one control propagates; changing α updates every linked exhibit.
- Cracktro Scroller: the method in focus comes forward; others recede in grey, never vanish.

## Unresolved

- Dark mode: not part of this world's first release; revisit after V7.
- Exact chapter order and copy for chapters 2–11 are drafted in V6–V7.
