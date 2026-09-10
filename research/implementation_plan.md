# Implementation plan

Status legend: [x] done · [~] in progress · [ ] not started. Updated as work lands;
`docs/HANDOFF.md` carries the live state.

## Framing

An **independent** project inspired by Berkeley URAP work on gender roles in news.
It does **not** use the Washington Post corpus that project used (not openly licensed).
The substitute is AmericanStories (Dell et al. 2023): article-level OCR of U.S.
newspapers digitized by the Library of Congress, CC-BY-4.0, public-domain source text,
1774–1963. We study **1900–1963**.

The deliverable is not one trend line. It is the trend **plus** evidence about whether
it survives different operationalizations of "gender signal", "person", and "role".

## Pipeline (each stage writes versioned files; nothing downstream is hand-entered)

| # | Stage | Code | Output |
|---|---|---|---|
| 1 | Ingest: stream per-year shards, sample articles | `src/gna/fetch.py` | `data/raw/articles_<year>.jsonl.gz`, `data/manifests/` |
| 2 | Prefix-sampling audit vs full-year download | `scripts/audit_prefix_sampling.py` | `results/sampling_audit.json` |
| 3 | Corpus quality: dupes (exact + MinHash), OCR quality, length drift, paper mix | `src/gna/quality.py` | `results/corpus_summary.parquet`, `research/corpus_quality_report.md` |
| 4 | Person extraction (spaCy NER + dependency parse), gender-signal hierarchy, rule roles, quotes, agency | `src/gna/extract.py` | `data/interim/entities.parquet`, `mentions.parquet` |
| 5 | Reference annotation sample (stratified) + annotation UI | `scripts/make_annotation_sample.py`, `site/…/annotate` | `data/annotation/` |
| 6 | Role methods A lexical · B dependency rules · C supervised BoW · D contextual-embedding classifier · E LLM (open-weights, optional) | `src/gna/roles/` | `results/method_labels.parquet` |
| 7 | Validation vs reference labels: P/R/F1, κ | `src/gna/validate.py` | `results/validation.parquet` |
| 8 | Topics (NMF, seed-stability checked) + page position as section proxy | `src/gna/topics.py` | `results/topic_model.json` |
| 9 | Temporal models: yearly shares, HAC trend, micro logit with topic/paper FE | `src/gna/models.py` | `results/role_trends.parquet`, `topic_adjusted.parquet` |
| 10 | Method agreement & disagreement over time | `src/gna/agreement.py` | `results/method_agreement.parquet` |
| 11 | Robustness grid (multiverse) | `src/gna/robustness.py` | `results/robustness.parquet` |
| 12 | Quotes & language (agency verbs, modifiers; FDR) | `src/gna/language.py` | `results/quote_trends.parquet`, `language.parquet` |
| 13 | Freeze: hashes of corpus, config, results, figures, paper | `scripts/freeze.py` | `results/release.json` |
| 14 | Figures | `scripts/figures.py` | `figures/*.png|svg` |
| 15 | Paper (numbers injected from results) | `scripts/build_paper.py` | `paper/paper.md` |
| 16 | Site (reads `results/` only) | `site/` | static Next.js export |

## Order of work (spec §56) and why

1. Prove **one** narrow measure is valid before scaling: the gender-signal hierarchy on
   person-in-article entities, validated on a reference sample.
2. Then one role construct (public-office / professional authority) across methods.
3. Only then topics, robustness, language extensions, site, paper.

## Compute tiers

- **smoke**: 3 years × 300 articles; runs in CI in under a minute (no network models).
- **standard**: 22 years (1900–1963, every 3 years) × ~10,000 articles.
- **extended**: same grid × 30,000 articles; larger encoders. Not run in CI.

## Known constraints

- Machine disk is ~97% full: shards are streamed, never stored.
- No human annotator is available in this build. Reference labels are produced by the
  AI build agent under a written protocol and are labelled as such everywhere
  (see `research/annotation_protocol.md`). Human re-annotation is the first
  listed next step, and the annotation UI exists to make it cheap.
