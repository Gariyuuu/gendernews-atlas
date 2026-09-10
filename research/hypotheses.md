# Pre-registered hypotheses

**Frozen:** 2026-09-10, before any person-mention, gender-signal, or role outcome was computed.
Corpus ingestion had run at this point; no NLP measurement had. The git commit that
adds this file is the freeze point — later edits go in the amendment log at the bottom,
never into the original wording.

These are hypotheses, not claims. Each will be marked supported / not supported /
untestable in `research/claims_registry.md` with a pointer to the generating script.

## Primary estimand

Among person mentions in U.S. newspaper articles (AmericanStories sample, 1900–1963)
that carry an **explicit textual gender signal**, the share signalled as women, and the
share of women-signalled mentions occurring in a given role context, as a function of
publication year.

Unit of analysis: the **person mention** (a PERSON entity span in an article), grouped
by article for inference (article-clustered standard errors). Article-level and
mention-level quantities are never mixed within one model.

## Hypotheses

**H1 — Visibility.** The share of gender-signalled person mentions that are signalled as
women increases between 1900 and 1963.

**H2 — Leadership/expert roles.** Conditional on a mention being classified into a
leadership, political, or expert/professional role, the share signalled as women is
lower than the all-mention share in every period, and the gap narrows over time.

**H3 — Composition.** Adjusting for inferred topic and newspaper (fixed effects)
changes the estimated 1900–1963 trend in H1 by a non-trivial amount (|Δ slope| of at
least 25% of the raw slope), i.e. topic/newspaper composition explains part of the raw trend.

**H4 — Method sensitivity.** Different role-measurement methods (rule-based lexical,
dependency-rule, supervised bag-of-words, contextual-embedding classifier) produce
materially different role-trend effect sizes: the ratio of largest to smallest
estimated slope for at least one headline role exceeds 2, or their signs disagree.

**H5 — Lexical exaggeration.** For roles measured both ways, the simple lexical method
yields a larger absolute trend than the context-aware method.

**H6 — Classifiability drift.** The fraction of person mentions that remain UNKNOWN
(no explicit gender signal) changes over time, and the H1 trend is sensitive to how
UNKNOWN is handled (excluded vs. bounded).

**H7 — Quotation.** Among attributed direct/indirect quotations with a gender-signalled
speaker, the share attributed to women is lower than their share of all
gender-signalled mentions ("seen more than heard", after Jia et al. 2016).

## Decisions fixed in advance

- Primary gender signal: honorific/title (Mrs., Miss, Mr., Madame, Lady, Sir, …) or
  gendered kinship/role noun in apposition, then an unambiguous pronoun in a short
  coreference window. **No first-name inference in the primary analysis**; a name-based
  variant may appear only as a labelled sensitivity analysis.
- UNKNOWN is a reported category, never silently dropped.
- Time is modelled continuously (year) with article-clustered SEs; annual points are
  never treated as i.i.d. observations for p-values.
- Families of role/verb/adjective tests use Benjamini–Hochberg FDR at q = 0.05; anything
  outside a pre-declared family is labelled exploratory.
- A headline claim is kept only if its sign survives every specification in the
  robustness grid; otherwise it is reported as tentative or unsupported.

## Amendment log

_(none yet)_
