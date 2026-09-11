# HANDOFF — GenderNews Atlas

_Last updated: 2026-09-10 (release 2dfc1b5 deployed). Every claim here was checked against
the repo or the live service when written. Update this file whenever state changes._

## Read this first

- **Canonical build: `~/Projects/gendernews-atlas-b`.** A second agent session wrote into
  `~/Projects/gendernews-atlas` concurrently (decision log D1). This build never modified
  it. The owner decides which to keep. The GitHub repo `Gariyuuu/gendernews-atlas` is
  pushed from `-b`.
- **The disk is shared and often near full.** Writers use `gna.paths.safe_write`. Every
  stage caches per year. Never delete other projects' caches.
- **`/tmp` gets wiped** (it happened mid-build). Keep logs and QA scripts somewhere
  durable, not in `/tmp`.
- **Local ports are shared with other sessions' servers.** Pick a free port (bind to 0) and
  assert the page title before trusting a screenshot.
- Use `.venv/bin/python`. Bare `python3` can resolve to another project's venv.

## Where things live

| What | Where |
|---|---|
| Live site | https://gendernews-atlas.vercel.app (Vercel project `gendernews-atlas`, SSO protection disabled) |
| Repo | https://github.com/Gariyuuu/gendernews-atlas (public; CI green on `2dfc1b5` and `25aadf5`) |
| Frozen release | `results/release.json` (hashes of results, figures, config; git sha) |
| Paper | `paper/paper.md`, built by `scripts/build_paper.py`; hand-written `paper/discussion.md` is spliced in |

## State

| Stage | State | Output |
|---|---|---|
| Corpus, 22 years, 220,221 articles | done | `data/manifests/`, raw text regenerable |
| Sampling audit | done, passes | `results/sampling_audit.json` |
| Quality, dedup, topics | done | `results/corpus_summary.parquet`, `results/topic_model.json` |
| Extraction (1,075,899 entities) + frame | done | `data/interim/` |
| Reference labels: 600 by AI annotator + 120 consistency | done | `data/annotation/` |
| Mixed-title rule (D17) | done; 1,679 clusters set to UNKNOWN | `data/interim/frame.parquet` |
| Trends, composition, language, validation (incl. by gender) | done | `results/*.parquet` |
| Methods C/D, fitted and applied to 110,000 people (109,518 after D17) | done | `data/interim/method_labels.parquet` |
| Method agreement, robustness, reviewer checks, claims | done | `results/method_agreement.parquet`, `robustness.parquet`, `claims.json` |
| Figures, paper, site | done and deployed | `figures/`, `paper/`, `site/dist/` |
| **LLM application subsample (200/year, year-interleaved)** | **in progress.** The deployed release uses the 266 people labelled before the gateway outage. The run was resumed and is year-balanced at any stopping point. | `data/interim/llm_apply.jsonl` |

Claims (pre-registered rules): H1 tentative, H2 unsupported, H3 supported, H4 supported,
H5 tentative, H6 tentative, H7 supported.

## Refresh after the LLM run grows (or finishes)

Only method E's numbers change. Re-run, in order:

```
.venv/bin/python scripts/analyze_methods.py      # checks E is a subset of the C/D sample
.venv/bin/python scripts/claims.py
.venv/bin/python scripts/figures.py
.venv/bin/python scripts/freeze.py && .venv/bin/python scripts/build_paper.py && .venv/bin/python scripts/freeze.py
.venv/bin/python scripts/export_site_data.py && .venv/bin/python site/build.py
git add -A && git commit && git push
cd site/dist && vercel link --yes --project gendernews-atlas && vercel deploy --prod --yes
rm site/dist/.env.local   # vercel link writes an OIDC token there; not served, not needed
```

Resuming the LLM run needs `GNA_LLM_API_KEY`, `GNA_LLM_BASE_URL=https://api.gariyuuu.com/v1`
and `GNA_LLM_MODEL="Yuu no Sekai"`, then `scripts/run_llm.py --mode apply --per-year 200
--workers 3`. It skips people who were already parsed.

## Deploys

- **Pushing does not deploy.** `site/dist` is gitignored because the site is built from
  pipeline outputs that Vercel cannot regenerate. Deploy by hand as above.
- The masthead hash reads `<sha>*` because the release files are written after the commit
  they describe. The release commit that follows (for example `25aadf5` for `2dfc1b5`)
  carries them.

## Bugs found and fixed (do not reintroduce)

- `analyze_composition`: `lpm_trend` returns its own `fe` key. Spreading `**r` after the
  spec name silently overwrote it. The spec name is now written last.
- `failure_examples`: the shared entity loader does not fetch `src_hp`; request it
  explicitly.
- LLM client: HTTP 429s at 6 workers. There is now exponential backoff with jitter; use
  2–3 workers. The gateway also went fully unresponsive for a while. Resuming is safe.
- Entity resolution merged couples ("Capt. Bissell … Mrs. Bissell") into one F entity with
  the husband's title (D17). The fix is `gna.frame.apply_mixed_title_rule`. Anything new
  that reads raw entity gender must apply it. `analyze_methods` checks the LLM subset
  against the sample as drawn, before the D17 drop.
- `run_llm.py --mode apply` used to walk year by year. It now interleaves years, so any
  prefix is year-balanced. The 79 calls from the old order are in
  `data/interim/llm_apply.prev-order.jsonl` and are not used.
- The resolution robustness summary compares only the aggregate grid, which runs under
  both rules. The fixed-effects and application cells run under the primary rule only.
  Every consumer that picks the "overall" summary row must also require `resolution` to be
  null.
- Share-positive labels are exact counts ("899 of 900"), never a percentage that rounds
  99.9% up to 100%.
- `node --test tests/` treats the directory as a module on Node 22+. Use the
  `tests/*.test.mjs` glob (package.json and CI do).

## Secrets

The method-E key lives only in the owner's memory notes and the `GNA_LLM_API_KEY` env var.
No key, token or `.env` file is in the repo (checked before each push).
