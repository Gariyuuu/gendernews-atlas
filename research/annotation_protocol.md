# Annotation protocol (reference labels, v1)

## Who annotates, stated plainly

The v1 reference labels were produced by **the AI build agent** (Claude Opus 5, the
same system that wrote this pipeline), following this written protocol. They are
**not human annotation** and **not expert annotation**, and are never described that
way anywhere in the project. They are called *reference labels*, not *gold*.

Consequences, stated up front:

1. Validation numbers measure agreement between the pipeline and a careful,
   protocol-bound AI reader. That is useful for separating methods from one another,
   but it is not a substitute for human validation.
2. Agreement between reference labels and the LLM condition (method E) is
   model-vs-model agreement and is flagged as such.
3. The intra-annotator consistency pass (below) measures *self-consistency*, not
   inter-annotator agreement.
4. **Human re-annotation of this sample is the first listed next step.** The
   annotation interface exists to make it cheap; human labels will replace v1 with no
   code changes.

## Blinding

The annotator sees only the context window, with the target mention highlighted, plus
year and newspaper. **No method outputs** (gender signal, roles A–E, quote flags) are
shown. The blind file (`data/annotation/sample_blind.jsonl`) is generated without
those columns, and labels are joined back by `entity_id` afterwards.

## Sample design

Stratified random sample of entities, seed fixed in `config/annotation.json`:
four periods (1900–15, 1918–30, 1933–45, 1948–63) × honorific class (F / M / none) ×
method-B role present (yes / no), with equal allocation per cell. Stratifying on a
method's own output is standard for rare-class recall estimation. Every estimate is
reweighted back to population proportions using the cell sampling fractions, and
unweighted numbers are never reported as population accuracy.

## Labels (per entity)

| Field | Values | Rule |
|---|---|---|
| `is_person` | yes / no / unsure | Is the highlighted span a reference to a human individual? (NER precision: OCR junk, place names, ship names, organisations and product names are "no".) |
| `gender_text` | F / M / UNKNOWN | Gender **signalled in the visible text only**: honorifics, pronouns clearly referring to this person, gendered nouns in apposition. First names are not evidence. When in doubt, UNKNOWN. |
| `roles` | subset of the 10 roles, or none | Only roles the text **attributes to this person**. A husband's office is not the wife's role. Organisational heads are coded by the organisation (club president → CIVIC). Past or honorary titles count ("former Senator"). |
| `quoted` | yes / no | Is speech (direct or reported) attributed to this person in the visible context? |
| `confidence` | 1 (guess) / 2 / 3 (certain) | |
| `notes` | free text | OCR problems, ambiguity, unusual constructions. |

### Edge cases decided in advance

- "Mrs. John Smith" → gender F (honorific). The given name belongs to her husband and
  is recorded nowhere as her name.
- Society-page lists ("Mrs. A, Mrs. B and Miss C were hostesses") → SOCIAL for each
  person the list covers.
- Obituaries: the deceased gets FAMILY only if identified through a kin relation;
  death by illness is not CRIME_ACCIDENT, death in a crash or by violence is.
- Advertising copy that names a person (endorsements) → roles from the text, if any.
- Sports: players, managers and coaches in a sports context → ARTS_SPORTS; "manager"
  outside sport → BUSINESS.
- Military rank on a civilian officeholder ("Col. Smith, the mayor") → both.

## Consistency pass

After the full sample is labelled, a random 20% is re-labelled blind in shuffled
order. Cohen's κ between the two passes is reported as **intra-annotator consistency**.
