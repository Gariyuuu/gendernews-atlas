# Limitations

Written to be read before the results. Each entry names the limitation, the direction
in which it can bias estimates where that is knowable, and what the project does about
it.

## Corpus

1. **Not a representative sample of American news.** AmericanStories covers what
   libraries chose to digitise under the National Digital Newspaper Program. Small-town
   weeklies and some states are over-represented, and major metropolitan dailies
   (other than the Washington *Evening Star* and a few others) are thin or absent.
   Results describe *this digitised corpus*, not the American press.
2. **The 1923 cliff.** Coverage falls sharply after 1922, when copyright begins to bind:
   about 270 newspapers per sampled year before, 37–106 after. The *Evening Star* rises
   from about 3% to about 57% of sampled articles. Any raw trend across 1923 mixes
   change in how people were written about with change in which newspapers are in the
   corpus. Mitigations: newspaper fixed effects, *Evening Star* exclusion, per-paper
   caps, pre/post-1923 split models, and a shift-share decomposition.
3. **Not the URAP corpus.** The Washington Post corpus used by the Berkeley URAP project is
   not openly licensed and was not used. This is an independent project on the same
   question; its numbers are not comparable to that project's.
4. **Sampling.** 22 years (every third year) × 10,000 articles. The streamed-prefix
   sampler was validated against a full-year download for 1963 and a deep stream for
   1900 only. Other years inherit the assumption that archive order is date-shuffled.

## Gender measurement

5. **Textual signal is not identity.** The measure is the gender signalled by period
   journalistic conventions. The honorific system (Mrs./Miss/Mr.) was binary and tied
   women's titles to marital status. People misgendered or erased by that system cannot
   be recovered from the text.
6. **Most people are UNKNOWN.** A large share of named people carry no honorific and no
   resolvable pronoun: people in lists, results, and legal notices. "Women's share"
   is therefore a share *of classifiable people*, and the UNKNOWN share is reported
   alongside it, with bounds.
7. **Differential classifiability by gender (likely upward bias in women's share).**
   Men in this press are often named by office or surname alone ("Senator Smith",
   "Jones said"), while women are almost always given a courtesy title. Men are
   therefore more likely to fall into UNKNOWN, which inflates women's share among
   classified people. The pronoun tier recovers some men. The bounds analysis shows how
   far the conclusion depends on this.
8. **The pronoun heuristic is a heuristic.** It uses nearest salient antecedent with a
   subject preference, not a learned coreference model. Its disagreement with honorifics
   is measured on every person who has both signals (`results/gender_consistency.parquet`).

## Extraction on historical OCR

9. **Modern models on old text.** spaCy `en_core_web_sm` was trained on contemporary
   web and news text. NER precision on 1900–1963 OCR is measured on the reference sample
   and is well below modern benchmarks. Non-person spans contaminate the UNKNOWN pool
   more than the gender-signalled pool.
10. **OCR noise is uneven.** The period-dictionary word rate and the legibility label are
    coarse proxies. They are used as a robustness filter, not a correction.
11. **Entity resolution is within-article and conservative.** Ambiguous surname
    mentions are dropped, and the same person across articles is counted once per
    article.

## Validation

12. **No human annotation.** Reference labels were produced by the AI build agent under
    a written protocol, blind to method outputs. They are neither human nor expert
    annotation. The intra-annotator consistency pass measures self-consistency only.
    Agreement between reference labels and the LLM condition is model-versus-model.
    Human re-annotation is the first next step.
13. **The reference sample is small** (hundreds of people across ten roles). Rare roles
    have few positives, so their F1 is imprecise and supervised methods C and D are
    weak there.

## Roles, topics and language

14. **The taxonomy is interpretive.** CIVIC and SOCIAL were added after the corpus
    profile (see the decision log). AUTHORITY was fixed before any outcome was
    computed. Role labels describe how the text positions a person, not the person's
    actual occupation.
15. **Topics are unstable in places.** Several NMF components do not replicate across
    seeds (society/weddings in particular), and topic labels were assigned by reading top
    terms. Composition adjustment is also run at the coarse section level.
16. **Language analyses are exploratory.** Many words are tested; FDR control is applied,
    but no word-level finding is a confirmed hypothesis.

## Inference

17. **22 time points.** Trend inference rests on few years. HAC standard errors with few
    points, and two-way clustering with 22 year clusters, are approximate.
18. **Linear probability models** are used for fixed-effects adjustment. Slopes are
    average changes in percentage points, not structural parameters.
19. **No causal claims.** Historical events are context, never explanations.

## LLM condition

20. **Hosted, versioned elsewhere.** Method E calls Qwen3-8B through the owner's gateway,
    which forwards to OpenRouter. The serving stack can change without notice, and one
    prompt was used. Prompt hash, model id and timestamps are recorded per call.

## Scope

21. **Race, class, region and non-English papers are not modelled.** The women most
    visible in this corpus (society pages, club news) were disproportionately white and
    affluent. Spanish- and German-language items are flagged by the topic model and
    excluded in the news-only specification.
