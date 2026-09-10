#!/usr/bin/env python
"""make models (method E) -- LLM labels on the reference sample and an application subsample.

Resumable: entity_ids already present in the output file are skipped.
Requires GNA_LLM_API_KEY (and optionally GNA_LLM_BASE_URL, GNA_LLM_MODEL).

  --mode reference   data/annotation/sample_blind.jsonl -> data/annotation/llm_reference.jsonl
  --mode apply       N per year from data/interim/method_labels.parquet -> data/interim/llm_apply.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.llm import LLMClient  # noqa: E402
from gna.paths import DATA, INTERIM  # noqa: E402


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["entity_id"] for x in open(path) if x.strip() and json.loads(x).get("parse_ok")}


def mark(ctx: str, s: int, e: int) -> str:
    return ctx[:s] + "⟦" + ctx[s:e] + "⟧" + ctx[e:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "apply"], required=True)
    ap.add_argument("--per-year", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    client = LLMClient()

    if a.mode == "reference":
        out = DATA / "annotation" / "llm_reference.jsonl"
        items = [json.loads(x) for x in open(DATA / "annotation" / "sample_blind.jsonl")]
        items = [{"entity_id": i["entity_id"], "year": i["year"], "publication": i["publication"], "text": i["text"]}
                 for i in items]
    else:
        out = INTERIM / "llm_apply.jsonl"
        ml = pd.read_parquet(INTERIM / "method_labels.parquet", columns=["entity_id", "year", "publication"])
        rng = np.random.default_rng(20260912)
        pick = pd.concat([g.iloc[np.sort(rng.choice(len(g), size=min(a.per_year, len(g)), replace=False))]
                          for _, g in ml.groupby("year")])
        ctx = pd.concat([pd.read_parquet(p, columns=["entity_id", "ctx", "ctx_hl_start", "ctx_hl_end"])
                         for p in sorted((INTERIM / "extract").glob("entities_*.parquet"))])
        pick = pick.merge(ctx, on="entity_id")
        items = [{"entity_id": r.entity_id, "year": int(r.year), "publication": r.publication,
                  "text": mark(r.ctx, int(r.ctx_hl_start), int(r.ctx_hl_end))} for r in pick.itertuples()]

    have = done_ids(out)
    todo = [i for i in items if i["entity_id"] not in have]
    print(f"{a.mode}: {len(items)} items, {len(have)} done, {len(todo)} to go", flush=True)
    n_ok = n_err = 0
    with open(out, "a") as fh:
        for k, rec in enumerate(client.label_many(todo, workers=a.workers), 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_ok += bool(rec.get("parse_ok"))
            n_err += not rec.get("parse_ok")
            if k % 100 == 0:
                print(f"  {k}/{len(todo)} ok={n_ok} err={n_err}", flush=True)
    print(f"done: ok={n_ok} err={n_err} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
