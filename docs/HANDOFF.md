# HANDOFF — GenderNews Atlas

_Last updated: 2026-09-10, mid-build. Every claim here was checked against the repo when
written. Update this file whenever state changes._

## Read this first

- **Canonical build: `~/Projects/gendernews-atlas-b`.** A second agent session wrote into
  `~/Projects/gendernews-atlas` concurrently (decision log D1). It had been idle for 20+
  minutes at last check. This build never modified it. The owner decides which to keep.
- **Disk is shared and often near 0 GB free.** Writers use `gna.paths.safe_write` (waits
  when free space is below `GNA_MIN_FREE_GB`, default 0.5; atomic rename; ENOSPC retry).
  Every stage caches per year. Never delete other projects' caches to make room.
- Use `.venv/bin/python`; bare `python3` on this machine can resolve to another project's venv.
- `>` onto a log file that orphaned workers still hold open mixes their tracebacks into
  the new log. A `BrokenPipeError` seen there came from a *killed* run, not the live one.

## State

| Stage | State | Output |
|---|---|---|
| Corpus: 22 years × ~10k articles (220,221) | done | `data/raw/` (gitignored), `data/manifests/` |
| Prefix-sampling audit | done, passes | `results/sampling_audit.json` |
| Corpus quality + dedup | done | `results/corpus_summary.parquet` |
| Topic model (NMF k=24 × 3 seeds) + labels | done | `results/topic_model.json`, `config/topic_labels.json` |
| Person extraction | running, resumable | `data/interim/extract/` |
| Reference sample → labels (AI annotator) → consistency pass | next | `data/annotation/` |
| Methods C/D, LLM E, validation | code ready | `scripts/role_methods.py`, `scripts/run_llm.py` |
| Analyses, robustness, claims | code ready | `scripts/analyze_*.py`, `robustness.py`, `claims.py` |
| Figures, freeze, paper, site | code ready | `scripts/figures.py`, `freeze.py`, `build_paper.py`, `site/build.py` |

## Run order after extraction

```
.venv/bin/python scripts/make_annotation_sample.py      # 600 blind items
# label data/annotation/sample_blind.jsonl -> labels_v1.jsonl (protocol), then the 20% consistency pass
GNA_LLM_API_KEY=... .venv/bin/python scripts/run_llm.py --mode reference
.venv/bin/python scripts/build_frame.py && .venv/bin/python scripts/analyze_trends.py
.venv/bin/python scripts/analyze_composition.py && .venv/bin/python scripts/analyze_language.py
.venv/bin/python scripts/role_methods.py                 # needs labels
GNA_LLM_API_KEY=... .venv/bin/python scripts/run_llm.py --mode apply
.venv/bin/python scripts/analyze_methods.py && .venv/bin/python scripts/robustness.py && .venv/bin/python scripts/claims.py
.venv/bin/python scripts/failure_examples.py && .venv/bin/python scripts/annotation_consistency.py
.venv/bin/python scripts/figures.py && .venv/bin/python scripts/freeze.py && .venv/bin/python scripts/build_paper.py
.venv/bin/python scripts/freeze.py   # again, so the paper hash is recorded
.venv/bin/python scripts/export_site_data.py && .venv/bin/python site/build.py
```

## Design decisions to keep

See `research/decision_log.md` (D1–D16). The load-bearing ones: textual gender only
(D6); honorific-class-aware entity resolution (D7); salience filter on pronouns (D8);
AUTHORITY fixed before outcomes (D9); AI reference labels disclosed (D11); LLM routing
disclosed (D12); zero-dependency site (D15); validated palette with no pink/blue (D14, D16).

## Secrets

The method-E key is in the owner's memory notes only and is passed as `GNA_LLM_API_KEY`.
It is never written to the repo.
