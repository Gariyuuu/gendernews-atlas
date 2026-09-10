#!/usr/bin/env python
"""make analyze (step 5) -- agency and descriptive language (exploratory, FDR-controlled).

* Agency: share of a person's syntactic positions that are AGENT (nsubj of a
  non-copular verb) vs PATIENT (nsubjpass / dobj), by signalled gender and year.
* Distinctive words: Monroe et al. (2008) log-odds with informative Dirichlet prior,
  F vs M, computed WITHIN each coarse section and pooled by inverse-variance weights
  (topic-controlled), with Benjamini-Hochberg FDR across each word family.

Output results/language.parquet (kinds: agency_yearly, agency_trend, logodds).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from gna.frame import LANG, load  # noqa: E402
from gna.models import bh_fdr, hac_trend, log_odds_dirichlet, yearly_share  # noqa: E402
from gna.paths import RESULTS, safe_write  # noqa: E402

MIN_COUNT = 25
SKIP = {"be", "have", "do", "-pron-", "who", "that"}


def counts(series: pd.Series) -> pd.Series:
    c = Counter()
    for lst in series:
        c.update(w for w in lst if w.isalpha() and w not in SKIP)
    return pd.Series(c, dtype=float)


def pooled_logodds(d: pd.DataFrame, col: str) -> pd.DataFrame:
    parts = []
    for sec, g in d.groupby("section"):
        a, b = counts(g.loc[g["f_hp"] == 1, col]), counts(g.loc[g["f_hp"] == 0, col])
        if a.sum() < 200 or b.sum() < 200:
            continue
        lo = log_odds_dirichlet(a, b)
        lo["section"] = sec
        parts.append(lo)
    if not parts:
        return pd.DataFrame()
    allp = pd.concat(parts).reset_index(names="word")
    allp["w"] = 1 / allp["var"]
    g = allp.groupby("word").agg(n_F=("n_a", "sum"), n_M=("n_b", "sum"), sw=("w", "sum"),
                                 swd=("delta", lambda s: 0.0), n_sections=("section", "nunique"))
    g["swd"] = allp.assign(wd=allp["w"] * allp["delta"]).groupby("word")["wd"].sum()
    g["delta"] = g["swd"] / g["sw"]
    g["z"] = g["delta"] * np.sqrt(g["sw"])
    g["p"] = 2 * stats.norm.sf(np.abs(g["z"]))
    g = g[(g["n_F"] + g["n_M"]) >= MIN_COUNT].copy()
    g["q"] = bh_fdr(g["p"].to_numpy())
    return g.drop(columns=["sw", "swd"]).reset_index()


def main() -> int:
    fr = load(["entity_id", "article_id", "year", "period", "section", "f_hp", "ner_ok"])
    lang = pd.read_parquet(LANG)
    d = fr[fr["ner_ok"] & fr["f_hp"].notna()].merge(lang, on="entity_id", how="left")
    for c in ("agent_verbs", "patient_verbs", "adjs", "poss_nouns"):
        d[c] = d[c].map(lambda x: list(x) if x is not None else [])
    d["n_agent"] = d["agent_verbs"].map(len)
    d["n_patient"] = d["patient_verbs"].map(len)
    d["n_pos"] = d["n_agent"] + d["n_patient"]
    rows = []

    pos = d[d["n_pos"] > 0].assign(agent_frac=lambda x: x["n_agent"] / x["n_pos"])
    for g, name in ((1.0, "F"), (0.0, "M")):
        s = yearly_share(pos[pos["f_hp"] == g], "agent_frac", weight="n_pos", B=300, seed=int(g))
        s["kind"], s["gender"] = "agency_yearly", name
        rows.append(s)
        t = hac_trend(s)
        rows.append(pd.DataFrame([{"kind": "agency_trend", "gender": name, **t}]))
    # share of entities appearing in any positioned role at all (visibility as actor/patient)
    for g, name in ((1.0, "F"), (0.0, "M")):
        s = yearly_share(d[d["f_hp"] == g].assign(_y=(d["n_pos"] > 0).astype(float)), "_y", B=200, seed=5 + int(g))
        s["kind"], s["gender"] = "positioned_yearly", name
        rows.append(s)

    for col, family in (("agent_verbs", "agent_verb"), ("patient_verbs", "patient_verb"), ("adjs", "modifier"),
                        ("poss_nouns", "possessed_noun")):
        for per in ["all"] + [str(p) for p in d["period"].cat.categories]:
            sub = d if per == "all" else d[d["period"].astype(str) == per]
            lo = pooled_logodds(sub, col)
            if lo.empty:
                continue
            lo["kind"], lo["family"], lo["period"], lo["stratification"] = "logodds", family, per, "within_section_pooled"
            rows.append(lo)
            print(f"{family:15s} {per:8s} words={len(lo):5d} q<.05: F={int(((lo['q'] < .05) & (lo['z'] > 0)).sum())} "
                  f"M={int(((lo['q'] < .05) & (lo['z'] < 0)).sum())}", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out["exploratory"] = True
    safe_write(lambda t: out.to_parquet(t, index=False), RESULTS / "language.parquet")
    lo = out[(out["kind"] == "logodds") & (out["period"] == "all")]
    for fam in lo["family"].unique():
        f = lo[lo["family"] == fam].sort_values("z")
        print(f"\n{fam}: most M-associated:", ", ".join(f.head(12)["word"]))
        print(f"{fam}: most F-associated:", ", ".join(f.tail(12)["word"][::-1]))
    print(out[out["kind"] == "agency_trend"][["gender", "slope_pp_dec", "ci_lo", "ci_hi", "level_1930"]].round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
