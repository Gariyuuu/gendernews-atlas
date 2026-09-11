# HANDOFF — GenderNews Atlas

_Last updated: 2026-09-10, late build. Every claim here was checked against the repo
when written. Update this file whenever state changes._

## Read this first

- **Canonical build: `~/Projects/gendernews-atlas-b`.** A second agent session wrote into
  `~/Projects/gendernews-atlas` concurrently (decision log D1). It was idle at last check.
  This build never modified it. The owner decides which to keep.
- **The disk is shared and often near 0 GB free.** Writers use `gna.paths.safe_write`.
  Every stage caches per year. Never delete other projects' caches.
- **Local ports are shared with other sessions' servers.** Port 8765 served someone else's
  app during QA. Pick a free port (bind to 0) and assert the page title before trusting a
  screenshot. The QA harness in the session scratchpad does both.
- Use `.venv/bin/python`. Bare `python3` can resolve to another project's venv.
- `>` onto a log that orphaned workers still hold open mixes their tracebacks into the new
  log.

## State

| Stage | State | Output |
|---|---|---|
| Corpus, 22 years, 220,221 articles | done | `data/manifests/`, raw text regenerable |
| Sampling audit | done, passes | `results/sampling_audit.json` |
| Quality, dedup, topics | done | `results/corpus_summary.parquet`, `results/topic_model.json` |
| Extraction (1,075,899 entities) + frame | done | `data/interim/` |
| Reference labels: 600 by AI annotator + 120 consistency | done | `data/annotation/` |
| LLM condition on reference sample | done, 600/600 | `data/annotation/llm_reference.jsonl` |
| Mixed-title rule (D17) in frame | done; 1,679 clusters set to UNKNOWN | `data/interim/frame.parquet` (`mixed_title`, `f_*_v2`) |
| Trends / composition / language | re-running on the D17 frame | `results/*.parquet` |
| Methods C/D fit + application | running | `data/interim/method_labels.parquet` |
| LLM application subsample (200/year, year-interleaved) | running, ~39 items/min | `data/interim/llm_apply.jsonl` |
| Validation re-run (`role_methods.py --skip-apply`) for D17 + by-gender rows, methods agreement, robustness, claims | next | — |
| Figures, freeze, paper, site | builders work; final run after the above | `figures/`, `paper/`, `site/dist/` |

## Bugs found and fixed this session (do not reintroduce)

- `analyze_composition`: `lpm_trend` returns its own `fe` key. Spreading `**r` after the
  spec name silently overwrote it. The spec name is now written last.
- `failure_examples`: the shared entity loader does not fetch `src_hp`; request it
  explicitly.
- LLM client: 267/600 first-run calls hit HTTP 429 at 6 workers. There is now an
  exponential backoff with jitter, and runs use 2–3 workers.
- Masthead: global `p { max-width }` caught the issue line, and uppercase styling turned the
  git hash into capitals.
- Entity resolution merged couples ("Capt. Bissell … Mrs. Bissell") into one F entity
  with the husband's title (D17). Fixed post hoc by `gna.frame.apply_mixed_title_rule`,
  applied in `build_frame`, validation, failure examples and method agreement. Samples
  drawn before the fix stay as drawn. Anything new that reads raw entity gender must
  apply the rule.
- `run_llm.py --mode apply` used to walk year by year, so a partial run covered only early
  years. It now takes a nested random order within each year and interleaves the years, so
  any prefix is year-balanced. The 79 calls from the old order are in
  `data/interim/llm_apply.prev-order.jsonl` and are not used.
- Robustness summaries gained a `resolution` column. Every consumer that picks the
  "overall" summary row must also require `resolution` to be null (`claims.py`,
  `build_paper.py` and `site/build.py` do).

## Secrets

The method-E key lives only in the owner's memory notes and the `GNA_LLM_API_KEY` env var.
