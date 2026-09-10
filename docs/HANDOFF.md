# HANDOFF — GenderNews Atlas

_Last updated: 2026-09-10 (build in progress). Every claim here was checked against
the repo at the time of writing. Update this file whenever state changes._

## Read this first

- **This directory (`~/Projects/gendernews-atlas-b`) is the canonical build.** A
  second agent session wrote into `~/Projects/gendernews-atlas` concurrently (see
  `research/decision_log.md` D1). That directory was never modified by this build.
  The owner decides which to keep.
- **The machine disk sits near 0 GB free** because other jobs share it. Every writer
  uses `gna.paths.safe_write` (wait + retry on ENOSPC). Every stage is resumable per
  year. Do not delete other projects' caches to make room.
- `python3` on this machine may resolve to another project's venv. Always use
  `.venv/bin/python`.

## What exists

| Stage | State | Where |
|---|---|---|
| Corpus: 22 years × ~10k articles (220,221) | done | `data/raw/` (gitignored, `make data` regenerates), manifests in `data/manifests/` |
| Prefix-sampling audit | done, passes | `results/sampling_audit.json` |
| Corpus quality + dedup | done | `results/corpus_summary.parquet`, `data/interim/articles_meta.parquet` |
| Person extraction | running / resumable | `data/interim/extract/entities_<year>.parquet` |
| Topic model (NMF k=24, 3 seeds) | queued after quality | `results/topic_model.json`; labels go in `config/topic_labels.json` |
| Reference annotation | not started | `scripts/make_annotation_sample.py` → `data/annotation/` |
| Methods C/D/E, validation | code written | `scripts/role_methods.py`, `scripts/run_llm.py` |
| Analyses | code written | `scripts/analyze_*.py`, `scripts/robustness.py` |
| Figures / freeze / paper / site | figures + freeze written; paper, site not started | `scripts/figures.py`, `scripts/freeze.py` |

## Run order

```
make data preprocess annotate      # annotate draws the sample; labels are added by hand/agent
make llm                            # optional, needs GNA_LLM_API_KEY
make models analyze robustness figures freeze paper site test
```

## Key decisions (details in research/decision_log.md)

Honorific + salience-filtered pronoun gender tiers, no first names (D6–D8). Entity
resolution is honorific-class-aware, so wives named by their husbands' names are never
merged with the husbands (D7). The taxonomy includes CIVIC and SOCIAL. Reference
labels come from the AI build agent and are disclosed as such (D11). The LLM goes via
the owner's gateway to OpenRouter (D12).

## Secrets

The method-E key lives only in the owner's memory notes and the `GNA_LLM_API_KEY`
environment variable. It is never written to this repo.
