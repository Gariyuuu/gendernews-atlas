#!/usr/bin/env python
"""make annotate -- draw the stratified reference-annotation sample.

Writes (committed; excerpts are public-domain source text):
  data/annotation/sample_blind.jsonl   what the annotator sees (NO method outputs)
  data/annotation/sample_key.csv       entity_id, stratum, population/sample counts, weight
  data/annotation/consistency_ids.txt  20% subset for the blind re-annotation pass
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.paths import DATA, INTERIM, load_config  # noqa: E402

ANN = DATA / "annotation"
ANN.mkdir(parents=True, exist_ok=True)


def period_of(year: int, periods) -> str:
    for a, b in periods:
        if a <= year <= b:
            return f"{a}-{b}"
    return "other"


def mark(ctx: str, s: int, e: int) -> str:
    return ctx[:s] + "⟦" + ctx[s:e] + "⟧" + ctx[e:]


def main() -> int:
    cfg = load_config("annotation")
    cols = ["entity_id", "article_id", "year", "publication", "hclass", "roles_b", "n_ner_mentions",
            "ctx", "ctx_hl_start", "ctx_hl_end"]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in sorted((INTERIM / "extract").glob("entities_*.parquet"))])
    df = df[df["n_ner_mentions"] >= 1].copy()
    df["period"] = df["year"].map(lambda y: period_of(int(y), cfg["periods"]))
    df["has_role_b"] = df["roles_b"].map(lambda r: "role" if len(r) else "norole")
    df["stratum"] = df["period"] + "|" + df["hclass"] + "|" + df["has_role_b"]

    rng = np.random.default_rng(cfg["seed"])
    picks = []
    for stratum, g in df.groupby("stratum", sort=True):
        n = min(cfg["per_cell"], len(g))
        idx = rng.choice(len(g), size=n, replace=False)
        s = g.iloc[np.sort(idx)].copy()
        s["pop_n"] = len(g)
        s["sample_n"] = n
        picks.append(s)
    samp = pd.concat(picks)
    samp["weight"] = samp["pop_n"] / samp["sample_n"]
    samp = samp.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)   # blind order

    with open(ANN / "sample_blind.jsonl", "w") as fh:
        for i, r in samp.iterrows():
            fh.write(json.dumps({"n": i + 1, "entity_id": r["entity_id"], "year": int(r["year"]),
                                 "publication": r["publication"],
                                 "text": mark(r["ctx"], int(r["ctx_hl_start"]), int(r["ctx_hl_end"]))},
                                ensure_ascii=False) + "\n")
    samp[["entity_id", "stratum", "period", "hclass", "has_role_b", "pop_n", "sample_n", "weight"]].to_csv(
        ANN / "sample_key.csv", index=False)
    k = int(round(cfg["consistency_fraction"] * len(samp)))
    cons = samp["entity_id"].sample(n=k, random_state=cfg["seed"] + 1).tolist()
    (ANN / "consistency_ids.txt").write_text("\n".join(cons) + "\n")
    print(f"sample={len(samp)} strata={samp['stratum'].nunique()} consistency={k} population={len(df)}")
    print(samp.groupby("stratum").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
