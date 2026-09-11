# Implementation plan

Status legend: [x] done · [~] in progress · [ ] not started (as of 2026-09-10, late build). Updated as work lands;
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
| 1 | [x] Ingest: stream per-year shards, sample articles | `src/gna/fetch.py`, `scripts/fetch_corpus.py` | `data/raw/articles_<year>.jsonl.gz` (gitignored), `data/manifests/` |
| 2 | [x] Prefix-sampling audit vs full-year download | `scripts/audit_prefix_sampling.py` | `results/sampling_audit.json` |
| 3 | [x] Corpus quality: dupes (exact + MinHash), OCR proxy, length, paper mix | `src/gna/quality.py`, `scripts/corpus_quality.py` | `results/corpus_summary.parquet`, `data/interim/articles_meta.parquet` |
| 4 | [x] Person extraction (spaCy NER + parse), gender-signal hierarchy, rule roles A/B, quotes, agency | `src/gna/extract.py`, `scripts/extract_persons.py` | `data/interim/extract/{entities,articles}_<year>.parquet` |
| 5 | [x] Reference sample (stratified, blind), AI-annotator labels, consistency pass, annotation UI | `scripts/make_annotation_sample.py`, `scripts/annotation_ingest.py`, `tools/annotate/` | `data/annotation/` |
| 6 | [~] Role methods C (bag-of-words) · D (MiniLM embeddings) · E (LLM, optional) | `src/gna/roles_ml.py`, `src/gna/llm.py`, `scripts/role_methods.py`, `scripts/run_llm.py` | `data/interim/method_labels.parquet`, `data/interim/llm_apply.jsonl` |
| 7 | [x] Validation vs reference labels: weighted P/R/F1; failure examples; self-consistency | `scripts/role_methods.py`, `scripts/failure_examples.py`, `scripts/annotation_consistency.py` | `results/validation.parquet`, `results/failure_examples.json`, `results/annotation_consistency.json` |
| 8 | [x] Topics (NMF k=24, 3 seeds, stability) + labels | `scripts/topics.py`, `config/topic_labels.json` | `results/topic_model.json`, `data/interim/article_topics.parquet` |
| 9 | [x] Temporal models: yearly shares, HAC trends, FE linear probability model (two-way clustered), adjusted yearly series, decomposition | `src/gna/models.py`, `scripts/analyze_trends.py`, `scripts/analyze_composition.py` | `results/role_trends.parquet`, `trend_summary.parquet`, `topic_adjusted.parquet` |
| 10 | [~] Method agreement & time-dependent disagreement | `scripts/analyze_methods.py` | `results/method_agreement.parquet`, `disagreement_examples.json` |
| 11 | [~] Robustness grid (multiverse) + reviewer checks | `scripts/robustness.py`, `scripts/reviewer_checks.py` | `results/robustness.parquet`, `reviewer_checks.parquet` |
| 12 | [x] Quotes & language (agency, log-odds, FDR) | `scripts/analyze_trends.py`, `scripts/analyze_language.py` | `results/quote_trends.parquet`, `language.parquet` |
| 13 | [~] Claims by pre-registered rule | `scripts/claims.py` | `results/claims.json`, `research/claims_registry.md` |
| 14 | [~] Freeze: hashes of corpus, config, results, figures, paper | `scripts/freeze.py` | `results/release.json` |
| 15 | [~] Figures (final pass after methods/robustness) | `scripts/figures.py` | `figures/*.png`, `*.svg` |
| 16 | [~] Paper (numbers injected from results) | `scripts/build_paper.py` | `paper/paper.md`, `research/corpus_provenance.md` |
| 17 | [~] Site: zero-dependency static build (decision D15), reads exported results only | `scripts/export_site_data.py`, `site/build.py`, `site/assets/` | `site/dist/` |

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
