# Measurement framework

Fixed before outcomes were computed (see `hypotheses.md` for the freeze commit).
Code references are to `src/gna/`.

## 1. Units of analysis

| Unit | Definition | Used for |
|---|---|---|
| article | one AmericanStories "full article" (headline + body, layout-segmented OCR) | topic, dedup, length, composition |
| mention | one person reference: a spaCy `PERSON` span, or (supplementary) a courtesy honorific + capitalised tokens that NER missed | robustness (mention-level shares) |
| **entity** | a **person-in-article**: mentions in one article resolved to the same individual | **all primary prevalence estimates** |
| speech event | a speech verb whose nominal subject is an entity mention | quotation analysis |

No model mixes units. Inference is clustered at the article (entity models) or at the
newspaper (composition models); time is never treated as i.i.d. annual points.

## 2. Text normalisation (`textnorm.py`)

Join line-break hyphenation; flatten newlines; normalise curly and OCR quote
artefacts (`''`, `,,`); repair doubled title periods (`Mr..`). **No spelling
correction**: modern correction models would inject modern regularities into
historical text.

## 3. Person mentions (`extract.find_mentions`)

- spaCy `en_core_web_sm` 3.8 NER `PERSON` spans.
- Titles are stripped from the span and from up to three contiguous preceding tokens.
  A coordinating "and" breaks the chain, so in "Dr. and Mrs. John Smith" only
  *Mrs.* attaches to the name.
- Discarded: spans with more than six name tokens, spans with no capitalised
  alphabetic token, and spans with no surname-like token of two or more letters.
- Supplementary `pattern` mentions (honorific + capitalised tokens not covered by NER)
  feed a robustness spec. **Primary analyses require at least one NER mention per entity.**

## 4. Entity resolution (`extract._cluster`)

1. Full-name mentions form entities keyed by *(courtesy-honorific class, given name,
   surname)*. Mentions with different honorific classes are **never merged**. That
   keeps "Mrs. Robert Jones" and "Senator Robert Jones", a period naming convention
   for spouses, as two people, so the husband's title cannot leak onto the wife.
2. A surname-only mention with an honorific ("Mrs. Jones") attaches to the unique
   same-class entity with that surname; if there is none, it attaches to the unique
   unclassed entity. Otherwise it is dropped as ambiguous and counted.
3. A surname-only mention without an honorific attaches to the unique entity with that
   surname. If several exist it is dropped as ambiguous; if none exist it starts a
   new entity.

The rules are symmetric in F and M.

4. **Mixed-title rule (decision D17, added after first results).** Rule 2's fallback
   lets "Mrs. Bissell" attach to the unique unclassed "Capt. Clayton L. Bissell", which
   is the couple construction and not one person. Any cluster that joins a female
   honorific (Mrs., Miss, Misses, Ms., Madame, Mme., Mlle., Lady) with any other title
   (name suffixes Jr./Sr. excepted) has every gender tier set to UNKNOWN
   (`gna.frame.apply_mixed_title_rule`). The pre-rule behaviour is robustness dimension
   `resolution = v2_merge`. Untitled couples are not detectable and remain an error
   source (limitations §11).

## 5. Gender-signal hierarchy (`extract.resolve_gender`)

The measured quantity is **the gender signalled by the text**, not a person's identity.

| Tier | Evidence | Spec name |
|---|---|---|
| 1 | Courtesy/noble honorific attached to any mention: *Mrs, Miss, Ms, Mme, Madame, Mlle, Lady, Dame, Queen, Princess, Duchess, Countess…* / *Mr, Messrs, Sir, Lord, King, Prince, Duke, Baron…* | `h` |
| 2 | Local pronoun: the first *he/him/his/himself* or *she/her/hers/herself* after a **salient** mention, within the same or next sentence, with no other person mention or human noun intervening. Entity-level: at least 2/3 agreement across mentions, otherwise UNKNOWN (conflict). | `hp` (**primary**) |
| 1b | Gendered nouns in apposition/compound (*widow, wife, actress, Sister, Father*) or kin titles | `hpn` (sensitivity only) |
| — | Nothing else → **UNKNOWN** | — |

- **Salience filter.** A mention cannot claim a following pronoun if its head is not an
  argument (prepositional object, possessor, compound: "wife of Senator Jones"), or if
  a different entity is the subject of the same sentence and this mention is not.
  This is a subject-preference heuristic after Lappin & Leass (1994).
- **Why gendered nouns are not primary.** Several of them (wife, widow, bride, hostess)
  are also FAMILY/SOCIAL role terms. Using them as gender evidence would make
  role × gender estimates partly true by construction.
- **Occupational "-man" compounds** (chairman, congressman, alderman) are role terms,
  never gender evidence; period usage applied them to women.
- **No first names are used as gender evidence** anywhere. French "M." is excluded
  because it cannot be told apart from an initial in OCR text.
- **Internal consistency check.** For entities with an honorific, the pronoun rule is
  still evaluated (`pron_signal`). Its agreement with the honorific gives a
  large-sample accuracy estimate for tier 2 at no annotation cost.
- **UNKNOWN is an analysis category.** Its share is reported by year, and the primary
  trend is re-estimated under bounds that assign all UNKNOWN to F, and to M.

## 6. Role taxonomy (multi-label, per entity)

| Role | Covers |
|---|---|
| PUBLIC_OFFICE | elected/appointed officials, diplomats, judiciary, police & law enforcement, candidates |
| MILITARY | officers and enlisted personnel (separate because of the 1918/1942/1945 wartime shocks) |
| BUSINESS | owners, executives, managers, merchants, bankers, brokers |
| LABOR | employees and wage workers, domestic service, farm and industrial work |
| PROFESSIONAL | medicine, science, education, law practice, clergy, journalism/authorship, experts |
| ARTS_SPORTS | performers, artists, athletes, coaches |
| CIVIC | officers and members of clubs, societies, churches, unions, parties, charities |
| SOCIAL | society-page roles: hostess, host, guest, debutante, bride, patroness |
| FAMILY | identified through a kin relation (*wife of, widow of, son of*) |
| CRIME_ACCIDENT | victims, suspects, defendants; subject of passive harm/arrest predicates |

**AUTHORITY** (pre-registered for H2) = PUBLIC_OFFICE ∪ BUSINESS ∪ PROFESSIONAL.
MILITARY and CIVIC are reported separately, not folded in.

The taxonomy follows the role list in the study brief plus two additions the corpus
profile made necessary: CIVIC (the women's-club press) and SOCIAL (society pages).
Neither was chosen to confirm a stereotype; both are where women-signalled mentions
concentrate, and omitting them would leave most of those mentions unclassifiable.

## 7. Role measurement methods

| Id | Method | Input | Notes |
|---|---|---|---|
| A | Lexical window | role nouns (NOUN/PROPN) within ±10 tokens in the same sentence, one sense per word | a fair, not strawman, baseline: POS-filtered |
| B | Dependency rules | titles on the name; appositives (either direction); copular / *elect·appoint·name* predicates; *serve as*; organisational heads (president, secretary, chairman…) disambiguated by the organisation they head; passive harm/arrest predicates | never inherits a role from another person's phrase |
| C | Supervised bag-of-words | logistic regression on position-marked context n-grams | trained on reference labels, cross-fitted |
| D | Contextual embeddings | sentence-transformer encoding of the entity context with the mention marked, plus a one-vs-rest logistic head | cross-fitted on reference labels |
| E | LLM (optional) | open-weights Qwen3-8B, served by OpenRouter behind the owner's gateway; JSON output, temperature 0, reasoning off | a measurement condition, never ground truth; model id, prompt hash and timestamp stored per call |

Supervised methods are trained and scored with K-fold cross-fitting on the reference
sample. Every entity's prediction comes from a model that did not see it.

## 8. Quotation (`extract.quote_events`)

A **speech event** is a token whose lemma is in a fixed speech-verb list and whose
`nsubj` is (a) a token of an entity mention (`how=name`), or (b) *he/she* resolved to
the nearest preceding salient entity of matching signalled gender within one sentence
(`how=pronoun`, secondary). It is **direct** if the sentence contains a quotation
mark, or if a short attribution fragment follows a quoted sentence. Quotation marks
alone are never counted without an attributed speaker.

## 9. Agency and description

For each mention head: `nsubj` of a non-copular verb → **agent** verb;
`nsubjpass` / `dobj` → **patient** verb; `poss` → possessed noun. Adjectival modifiers
(`amod`) of the name or of an appositive head → **modifiers**. Group contrasts use
weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008),
within topic strata, with Benjamini–Hochberg FDR at q = 0.05.

## 10. Composition covariates

- **Newspaper** (normalised title / LCCN).
- **Topic**: NMF on article TF-IDF. Components are labelled by inspecting top terms
  and are checked for stability across seeds. Page 1 vs. interior is a coarse section
  proxy; AmericanStories carries no section field.
- **Article length** (words). Counts are always expressed as shares or per-10k-word rates.
- **Duplicates**: exact hash and MinHash-LSH (5-word shingles, Jaccard ≥ 0.8) within year.

## 11. Temporal model

- Yearly shares with article-cluster bootstrap 95% CIs.
- Trend: OLS of yearly share on year, weighted by entity count, with Newey–West HAC
  standard errors (lag 2), reported as **percentage points per decade**.
- Composition-adjusted trend: entity-level logistic regression with year plus topic
  and newspaper fixed effects, SEs clustered by newspaper. Reported as the average
  marginal effect in pp/decade and compared with the same model without fixed effects,
  so the raw-vs-adjusted comparison is within one model family.

## 12. Robustness grid (multiverse)

gender tier {h, hp, hpn} × role method {A, B, C, D, (E)} × entity filter {NER-only,
+pattern} × dedup {keep, drop near-dups} × OCR {all, legible & dict-rate ≥ 0.75} ×
newspapers {all, excl. *Evening Star*, cap per paper-year} × time bins {year, 5-yr,
decade} × classifier threshold {0.3, 0.5, 0.7}. A headline claim is kept only if its
sign survives every cell.
