# GenderNews Atlas

**How robust are NLP measurements of changing gender representation in U.S. news, 1900–1963?**

GenderNews Atlas measures how American newspapers represented people signalled as women
and as men between 1900 and 1963. It then tests whether those conclusions survive a
change of NLP method: five role-measurement methods, three gender-evidence rules,
composition adjustment for topic and newspaper, and a multiverse of analysis choices.

It is an independent project inspired by Berkeley URAP research on gender roles in
news. It does **not** use that project's Washington Post corpus, which is not openly
licensed. It uses [AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
(Dell et al. 2023, CC-BY-4.0), article-level OCR of newspapers digitised by the Library
of Congress.

## Where the results are

No number in this README is typed by hand. The results live in generated files:

| What | Where |
|---|---|
| Pre-registered hypotheses (frozen before measurement) | `research/hypotheses.md` |
| Claim statuses (supported / tentative / unsupported), computed by rule | `research/claims_registry.md`, `results/claims.json` |
| Headline numbers + provenance hashes | `results/release.json` |
| Result tables | `results/*.parquet` |
| Figures | `figures/` |
| Manuscript | `paper/paper.md` |
| Site | `site/dist/` after `make site` |

## Measurement in one paragraph

The unit is the person-in-article entity. Gender comes only from **textual** evidence:
courtesy honorifics, then a salience-filtered local pronoun rule. UNKNOWN is kept as a
category, and first names are never used. Roles are measured by (A) a lexical window,
(B) dependency rules, (C) a supervised bag-of-words model, (D) a contextual-embedding
classifier and (E) an open-weights LLM, all on the same people. Every method is
validated against a stratified reference sample. **The reference labels were produced by
an AI annotator, not humans**; see `research/annotation_protocol.md`. A human annotation
tool is in `tools/annotate/`.

## Reproduce

```
make setup                  # uv venv (Python 3.11) + pinned requirements + spaCy model
make data                   # stream + sample the corpus (no tarball stored)
make preprocess             # quality/dedup, extraction, topics
make annotate               # draw the blind reference sample
make llm                    # optional: needs GNA_LLM_API_KEY
make models analyze robustness figures freeze paper site
make test                   # python + chart-kit tests
```

`TIER=smoke` runs a three-year, 300-article-per-year version in minutes. All intermediate
stages are cached per year and resumable.

## Layout

```
src/gna/        fetch, text normalisation, extraction, quality, models, roles_ml, llm, frame
scripts/        one script per pipeline stage (see Makefile)
config/         corpus, annotation, topic labels, palette, period dictionary
research/       plan, questions, hypotheses, literature, measurement framework, protocol,
                claims, limitations, ethics, decision log, reviews
data/manifests  per-year manifests with content hashes (raw text is regenerated, not committed)
data/annotation reference sample (blind file, key, labels)
site/           zero-dependency static site (build.py + vanilla SVG chart kit)
tools/annotate  standalone annotation interface
docs/HANDOFF.md current state for the next session
```

## Integrity notes

- The hypotheses were committed before any measurement (`git log research/hypotheses.md`).
- Claims are kept only if their sign survives every robustness cell; the rest are
  reported as tentative or unsupported, not dropped.
- Limitations (`research/limitations.md`) include an expected upward bias in women's
  share among classified people. Men are more often named without honorifics, so more
  men fall into UNKNOWN.

## Licence and attribution

Code: MIT. Corpus: AmericanStories, CC-BY-4.0 (Dell, Carlson, Bryan, Silcock, Arora,
Shen, D'Amico-Wong, Le, Querubin, Heldring 2023), from Chronicling America (Library of
Congress). Period dictionary: Webster's Second International (1934), public domain.
