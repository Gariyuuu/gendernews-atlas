# Decision log

Chronological, append-only. Each entry records the decision, the alternatives
considered, and why. Dates are UTC.

## 2026-09-09

**D1 — Build directory `gendernews-atlas-b`, not `gendernews-atlas`.**
A second agent session on this machine received the identical brief and wrote into
`~/Projects/gendernews-atlas` while this build was starting there. It deleted a
source file this build had written. The question to the owner failed to send (the
tool stream closed), so this build moved to an isolated directory. The owner picks
which copy to keep; nothing in the other directory was modified.

**D2 — Corpus: AmericanStories (Dell et al. 2023), 1900–1963.**
Considered: the Washington Post corpus used by the URAP project (not openly
licensed, so not used); the LOC Chronicling America API (the legacy
`chroniclingamerica.loc.gov` search API now returns 404, and `loc.gov` JSON gives
metadata but no OCR text); GDELT (no full text; starts 1979/2013); CC-News (2016+
only). AmericanStories is article-segmented, CC-BY-4.0, built on public-domain
source text, and covers 1774–1963.

**D3 — Streamed-prefix sampling, validated.**
Year shards are 0.4–7 GB. Tar members are date-shuffled, so the pipeline streams
each shard and stops at 10,000 articles. That this approximates random sampling is
*tested*, not assumed (`scripts/audit_prefix_sampling.py`). Against a complete 1963
download of 51,600 scans, prefix TVDs for newspaper, month and page all fall inside
the random-sample null (one-sided p = 0.48 / 0.23 / 0.35). A deep stream of 1900
shows no ordering drift.

**D4 — Year grid 1900–1963, every three years (22 years).**
It trades temporal resolution against parse cost, and gives 22 time points for
trend models with HAC errors.

## 2026-09-10

**D5 — Hypotheses frozen before measurement** (commit `720e804`).

**D6 — Primary gender signal = honorifics + salience-filtered local pronouns.**
Gendered nouns (wife, widow, actress…) are a sensitivity tier only, because several
are also role terms and would make role × gender estimates partly mechanical. No
first names, anywhere. French "M." is excluded because it collides with initials.

**D7 — Honorific-class-aware entity resolution.**
The first design clustered mentions by surname. That would merge "Mrs. Robert Jones"
with "Senator Robert Jones" (the period convention of naming wives by their husbands'
names) and hand the Senator's office to the wife. Mentions with different courtesy
classes now never merge, and surname-only mentions attach only when the attachment
is unique. A regression test covers the case.

**D8 — Subject-salience filter on the pronoun tier.**
The smoke test showed "She said…" being attributed to "Senator Robert Jones", who
sits inside "wife of Senator Robert Jones", instead of to the sentence subject, Mary
Jones. A mention in a non-argument position, or a non-subject mention when another
entity is the subject, can no longer claim a following pronoun (after Lappin & Leass
1994). Regression-tested.

**D9 — Taxonomy adds CIVIC and SOCIAL; MILITARY separate from PUBLIC_OFFICE.**
The corpus profile showed women-signalled mentions concentrated in club and society
news. Without these categories, most such mentions would be role-less. MILITARY is
kept apart so wartime years (1918, 1942, 1945) do not swamp PUBLIC_OFFICE.
AUTHORITY = PUBLIC_OFFICE ∪ BUSINESS ∪ PROFESSIONAL, fixed before outcomes.

**D10 — Method A is a fair baseline, not a strawman.**
Its lexical window is POS-filtered (nouns and proper nouns). An unfiltered window
would inflate A's errors ("general election", "private") and make H5 easy to confirm.

**D11 — Reference labels by the AI build agent, disclosed.**
No human annotator was available. Labels come from the agent under a written
protocol, blind to method outputs. They are called "reference labels", never "gold",
"human" or "expert". A human-annotation UI (`tools/annotate/`) makes replacement cheap.

**D12 — Method E routing disclosed.**
The owner's gateway (`api.gariyuuu.com`, model "Yuu no Sekai" = Qwen3-8B) proxies to
OpenRouter. An earlier draft of `ethics.md` wrongly called this local inference; it
was corrected before any LLM call on corpus text.

**D13 — Disk-full resilience.**
The machine's disk is shared with other jobs and repeatedly hit 0 GB free, which
killed a run with ENOSPC. Writers now go through `safe_write` (wait for space, atomic
rename, retry), and year-level caching makes every stage resumable. Other projects'
files and caches were not touched.

**D15 — Zero-dependency static site instead of Next.js.**
With the machine at about 0.1 GB free, `npm install` for a Next.js app (hundreds of MB
of `node_modules`) was not possible. The site is a Python build script
(`site/build.py`) that renders every route to static HTML from the frozen results.
Headline numbers are pre-rendered into the markup, so the page reads without
JavaScript. A single vanilla ES module draws the SVG charts from the same JSON, with
hover tooltips and table views. This matches the portfolio's zero-build docs-site
pattern and deploys to Vercel as plain files. Consequence: the brief's "TypeScript
check" CI step becomes `node --check` plus `node:test` unit tests on the chart module
(built into Node, no dependencies).

**D14 — Chart colours: women vermilion, men violet, no pink/blue.**
Validated with the dataviz six-check validator in both modes (worst CVD ΔE 29.5
light / 26.0 dark). Every series also carries a dash pattern (the portfolio
MASTER.css redundant-encoding rule). Method comparisons use small multiples in one
hue, not a five-colour palette.

**D16 — Site theme and the vermilion step.**
Hallmark diversification: the portfolio's recent builds used Long Document + Newsprint +
N6 + Ft4 for a research page, so this site takes Stat-Led, a custom theme (light paper,
roman-serif display, cool ink-blue UI accent), N6 masthead and Ft1 footer. It differs
from the last build (Almanac, warm) on accent hue. The hero figure is computed from
`results/release.json`, never typed. Re-validating the chart pair against the real site
paper (`#f9f6f1`) showed the first women step at 2.97:1, below the 3:1 mark floor. It
was snapped to the next validated step (`#d95926`: 3.60:1, CVD ΔE 27.3), and figures and
site share it via `config/palette.json`.

**D17 — Mixed-title clusters lose their gender signal (post-results measurement fix).**
Inspecting women coded MILITARY by method B showed couple constructions ("Capt.
Clayton L. Bissell … Mrs. Bissell", "Vice Admiral Edward L. Cochrane and Mrs.
Cochrane") forming one entity. The surname-only *Mrs.* mention attaches to the unique
neutral-class full name. This is the path in `extract._cluster` meant for "Jane Addams …
Miss Addams". The merged entity is labelled F and carries the husband's title. Among
NER-supported entities, 1,370 of 175,078 F entities (0.8%) join a female honorific with
another title. They are 13.3% of F entities in method-B authority roles, falling from
23.7% (1900–15) to 9.6% (1948–63), so they inflate the level of H2 more early than
late.

Fix: `gna.frame.apply_mixed_title_rule` sets every gender tier to UNKNOWN on such
clusters. Titles are stored per entity, so no re-extraction is needed, and entity ids
(and so the reference labels) are unchanged.

The rule was added after the first results were seen, so it is a deviation from the
frozen plan. The old behaviour stays in the multiverse as `resolution = v2_merge`. The
validation reports the gender rule both ways (`gender_hp_v2` and `gender_hp`). Samples
drawn before the rule (the reference sample, the C/D application sample and the LLM
subsample) are left as drawn, and mixed clusters are removed at analysis time. Untitled
couples ("Clayton Bissell … Mrs. Bissell") cannot be detected this way (see
limitations).
