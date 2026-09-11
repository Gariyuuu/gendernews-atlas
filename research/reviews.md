# Mock peer review and author response

Three reviews were written against the frozen design and results, as the brief's
"Reviewer 2" step requires, followed by a point-by-point response. Where a fix was
feasible it was implemented, and the response says where to find it. Numbers are not
repeated here: they live in `results/` and `research/claims_registry.md`, which are
regenerated from code.

---

## Review 1 — supportive

The paper treats measurement as the object of study rather than a preliminary, which is
the right move for a literature that mostly reports one pipeline's trend. Strengths:

1. Hypotheses were frozen and committed before any NLP outcome existed, and claim
   statuses are computed by rule.
2. Gender is assigned only from textual evidence. UNKNOWN is reported, not dropped, and
   bounded.
3. Corpus composition (the 1923 digitisation cliff, the *Evening Star*'s growth) is
   modelled with fixed effects, sample restrictions and a shift-share decomposition, not
   footnoted.
4. Five role methods are compared on the same people, and each is validated against a
   stratified, sampling-weighted reference sample.
5. The limitations section is unusually candid, including a directional prediction
   (upward bias in women's share among classified people) that the reference labels then
   bear out.

Requests: (a) human validation of at least a subsample; (b) a learned coreference
comparison for the pronoun tier; (c) extension past 1963 with a licensed corpus.

## Review 2 — skeptical

I am not persuaded that the headline quantities measure what the prose implies.

1. **Binary gender.** The categories come from Mrs./Miss/Mr. That is a period convention,
   not a theory of gender. The paper must not let "women's share" drift into a claim
   about women.
2. **Identity inference.** The LLM condition is the standard route to name-based
   inference by the back door. How much of its "gender" comes from first names?
3. **Copyright.** Are excerpts redistributed? Under what licence?
4. **Representativeness.** This is the digitised record, dominated late by one
   Washington paper. "American newspapers" in the title question overreaches.
5. **Historical language change.** Role lexicons are fixed across 64 years. "Secretary",
   "chairman" and "president" changed meaning and gender association. A fixed lexicon
   will manufacture trends.
6. **LLM bias.** The strongest method on role F1 is the LLM, validated against labels
   produced by *another* LLM, the build agent. That is model-versus-model agreement
   and says nothing about truth.
7. **Method disagreement.** If methods disagree, which trend is right? A disagreement
   table is not an answer.
8. **Selection bias.** Who gets named, and who gets an honorific, is itself gendered.
   Men named by office alone fall into UNKNOWN. The "share of women among
   gender-signalled people" is then largely a measure of honorific usage.
9. **Multiple comparisons.** Ten roles × two methods × three tiers × many
   specifications: something will be significant.
10. **Causal overclaiming.** Suffrage in 1920 and wartime employment will tempt readers
    into causal stories. The design supports none.
11. **Circular validation.** The same AI system built the pipeline and produced the
    reference labels. The self-consistency κ is near 1 because the annotator remembers
    its own answers.
12. **Extraction quality.** Roughly three in ten extracted "people" in the reference
    sample are not people. How does that noise distribute across the measures?

## Review 3 — methods

1. **Few clusters.** Year-level shocks with 22 years make analytic cluster SEs
   unreliable. Use a wild-cluster bootstrap.
2. **LPM for a binary outcome.** Show that a logit gives the same answer.
3. **FE identification.** With newspaper fixed effects, identification comes from
   within-paper change over time. Late years are dominated by one paper, so the adjusted
   slope is largely an *Evening Star* slope. Say so and show the excluding-*Star*
   estimate.
4. **HAC with 22 points.** Newey–West with so few points is fragile. Report the
   micro-model alongside it, and do not rest a claim on one standard error.
5. **Bootstrap design.** Article-cluster resampling ignores newspaper-level dependence
   in the yearly-share CIs.
6. **Topic instability.** Several NMF components do not replicate across seeds. Do the
   composition results depend on topic granularity?
7. **Classifier training.** About 550 reference people, ten multi-label roles, and
   stratified sampling. Were sampling weights used in training? How do thresholds move
   the trends?
8. **Specification curve.** Cells are weighted equally, but the grid is not a random
   sample of plausible analyses. "Share of cells positive" is descriptive only.
9. **Multiple testing.** Control FDR across the role-trend family.

---

## Author response

Changes made in response are marked **[implemented]**. Points we accept but cannot fix
within this build are marked **[limitation]**.

**R2.1 Binary gender.** Agreed on the framing. Every page and the paper call the
quantity "gender signalled in the text". The ethics statement says the categories
reflect a binary, marital-status-marked honorific system. UNKNOWN is a reported category
with bounds. **[implemented: wording]** **[limitation: the text cannot recover
identities it erased]**

**R2.2 Identity inference.** Measured rather than assumed. On reference people whose
text gives no gender signal, the LLM still assigns a gender to most of them, skewing
male (`results/validation.parquet`, row `gender_assigned_without_text_signal`). The LLM's
gender output is therefore **never used in any estimate**; it appears only as a
validation row. The rule-based tiers assign a gender to text-silent people in a small
fraction of cases, mostly through pronoun errors. **[implemented]**

**R2.3 Copyright.** The sources are pre-1964 U.S. newspapers in the public domain; the
dataset is CC-BY-4.0 and attributed. The repository holds derived data and short
excerpts; raw sampled text is regenerated by `make data`. The site has no search and no
text API.

**R2.4 Representativeness.** Accepted. The site and paper describe "the digitised
corpus", not "American newspapers". Composition is handled four ways: fixed effects,
excluding the *Evening Star*, per-paper caps, and pre/post-1923 splits. **[implemented:
wording]** **[limitation]**

**R2.5 Language change.** Partly addressed by design. Method B types organisational heads
by the organisation they head, so "president of the club" is CIVIC rather than public
office. Method D reads context rather than a lexicon. Methods A–D differ most on exactly
the roles where drift is plausible (`results/method_agreement.parquet`, kind `h4`).
**[limitation: no diachronic lexicon]**

**R2.6 LLM bias and model-versus-model validation.** Accepted in full. Every validation
number is labelled "reference labels are AI-produced, not human". LLM–reference agreement
is flagged as model-versus-model in the protocol, the paper and the site. Human
re-annotation is the first listed next step, and a zero-dependency annotation tool
(`tools/annotate/`, also at `/annotate/` on the site) exports labels in the pipeline's
schema. **[limitation; tooling implemented]**

**R2.7 Which method is right?** None is privileged. The pre-registered rule keeps a claim
only if its sign holds under every method and specification. Otherwise it is reported
as tentative or unsupported. The disagreement explorer shows the passages where methods
split, so readers can judge.

**R2.8 Selection into gender signalling.** This is the central measurement finding, not a
threat we missed. It was predicted in `limitations.md` before results. The reference
labels confirm it: an honorific-only rule recovers most women but a minority of men
(`results/validation.parquet`, `gender_h` rows). The paper therefore treats *levels* of
women's share as uninterpretable and focuses on *changes*, the UNKNOWN share, and the
bounds. **[implemented: interpretation]**

**R2.9 Multiple comparisons.** Benjamini–Hochberg q-values across the role-trend family
(`results/reviewer_checks.parquet`, check `bh_fdr_role_trends`). Language analyses were
FDR-controlled and labelled exploratory from the start. Headline claims are limited to
seven pre-registered hypotheses. **[implemented]**

**R2.10 Causal overclaiming.** No causal claims are made. The shift-share decomposition is
labelled an accounting identity. Historical events appear only as annotations of the
digitisation cliff. **[implemented: wording]**

**R2.11 Circular validation.** Accepted, and disclosed as above. The self-consistency
statistic is explicitly an upper bound. The site states that a near-perfect κ is
expected for that reason and is not evidence of correctness. **[limitation]**

**R2.12 Extraction noise.** The NER precision estimate is reported on the validation and
failures pages. Non-person spans rarely carry honorifics, so they mostly enter UNKNOWN,
which inflates the UNKNOWN share more than it biases women's share among classified
people. The failures page shows examples. **[limitation]**

**R3.1 Few clusters.** Wild-cluster restricted bootstrap over the 22 year clusters for
the headline raw and adjusted slopes (`results/reviewer_checks.parquet`, check
`wild_cluster_bootstrap_year`). The raw trends are significant with year clusters alone.
The composition-adjusted slopes are not distinguishable from zero under either the wild
bootstrap or analytic two-way clustering. **[implemented]**

**R3.2 LPM vs logit.** Logit average marginal effects reproduce the LPM slopes almost
exactly, with and without topic fixed effects (check `logit_vs_lpm`). **[implemented]**

**R3.3 FE identification.** Accepted. The adjusted results are reported alongside the
excluding-*Evening Star* sample. The two differ, and the paper states that the
all-paper adjusted slope is heavily weighted by one paper's within-paper change
(`results/topic_adjusted.parquet`). **[implemented: reporting]**

**R3.4 HAC with few points.** Every headline trend is reported under both the HAC
yearly-series estimator and the micro LPM with two-way clustered SEs. Claims rest on
sign survival across the multiverse, not on either standard error. **[implemented]**

**R3.5 Bootstrap design.** The yearly-share CIs are article-cluster CIs and are labelled as
such. Newspaper-level dependence enters the trend inference through the two-way
clustered LPM and the wild-cluster checks. **[partly implemented]**

**R3.6 Topic instability.** The decomposition is reported by topic and by coarse section.
The composition and within components are close under both groupings
(`results/topic_adjusted.parquet`, kind `decomposition`). **[implemented]**

**R3.7 Classifier training.** C and D are trained with sampling weights, so their
probabilities target population prevalence. They are cross-fitted, so no reference item
is scored by a model that saw it. The robustness grid varies thresholds over 0.3, 0.5
and 0.7. With about 550 labelled people, rare roles have too few positives and are
reported as weak. **[implemented; limitation for rare roles]**

**R3.8 Specification curve.** Agreed. "Share of cells positive" is presented as a
description of the grid, not a probability. The decision rule is sign survival in *every*
cell, which does not depend on weighting. **[implemented: wording]**

**R3.9 FDR.** See R2.9. **[implemented]**

## Post-review audit (found by the authors, not raised by a reviewer)

**A1 Couple clusters.** Reading raw cases behind a surprising number (women coded
MILITARY by method B) turned up a resolution error. In "Capt. Clayton L. Bissell … Mrs.
Bissell" the surname-only *Mrs.* joins the husband's titled entity, which is then labelled
a woman holding his title. These clusters are 13% of women in B-coded authority roles, more
in early years. Fix: decision D17. Such clusters lose their gender signal, and the old
behaviour is a robustness dimension. Because the fix came after results were seen, the
paper labels it as a deviation. **[implemented]**

**A2 Differential error by gender.** The same audit showed method B's authority
precision to be much lower among women than among men: club and church officers are read
as holders of office. Validation now reports authority precision and recall separately by
signalled gender for every method, and the paper states the direction of the resulting
bias in H2's level. **[implemented: disclosure; no correction attempted at this sample
size]**
