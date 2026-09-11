#!/usr/bin/env python
"""make analyze (step 4) -- method comparison and disagreement on one shared population.

Every method is evaluated on the SAME entities: the application subsample (up to
5,000 gender-signalled NER entities per year) for A/B/C/D, and its LLM sub-subsample
for E.  Nothing here compares methods on different populations.

Outputs
  results/method_agreement.parquet    kinds: yearly (prevalence & female share by method),
                                      trend (HAC slope by method), kappa (pairwise, by period),
                                      h4 (slope spread per role), kappa_trend (disagreement over time)
  results/disagreement_examples.json  inspectable cases where methods disagree
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from gna.frame import ROLE_COLS, is_f, load, role_flags  # noqa: E402
from gna.lexicons import AUTHORITY, ROLES  # noqa: E402
from gna.models import hac_trend  # noqa: E402
from gna.paths import INTERIM, RESULTS, safe_write  # noqa: E402

TH = 0.5


def method_flags(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"A": role_flags(m["roles_a"], "A").rename(columns=lambda c: c[2:]),
           "B": role_flags(m["roles_b"], "B").rename(columns=lambda c: c[2:])}
    for pre in ("C", "D"):
        for th in (0.3, 0.5, 0.7):
            f = pd.DataFrame({r: m[f"{pre}_{r}"] >= th for r in ROLES}, index=m.index)
            f["AUTHORITY"] = f[list(AUTHORITY)].any(axis=1)
            out[pre if th == TH else f"{pre}@{th}"] = f
    return out


def load_llm(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    rec = [json.loads(x) for x in open(path)]
    rec = [r for r in rec if r.get("parse_ok")]
    if not rec:
        return None
    e = pd.DataFrame({"entity_id": [r["entity_id"] for r in rec],
                      "E_roles": [r["parsed"]["roles"] for r in rec],
                      "E_gender": [r["parsed"]["gender_text"] for r in rec],
                      "E_is_person": [r["parsed"]["is_person"] for r in rec]})
    return e


def main() -> int:
    m = pd.read_parquet(INTERIM / "method_labels.parquet")
    # the sample was drawn on the v2 gender signal; mixed-title clusters lose it (decision D17)
    m = m.merge(load(["entity_id", "mixed_title"]), on="entity_id", how="left")
    mixed = m["mixed_title"].fillna(False).astype(bool)
    print(f"application sample: {len(m)} entities, {int(mixed.sum())} dropped as mixed-title clusters", flush=True)
    app_ids = set(m["entity_id"])                 # the drawn sample, before the D17 drop
    m = m[~mixed].reset_index(drop=True)
    m["f"] = is_f(m["gender_hp"])
    m["period"] = pd.cut(m["year"], [1899, 1915, 1930, 1945, 1963], labels=["1900-15", "1918-30", "1933-45", "1948-63"])
    flags = method_flags(m)
    llm = load_llm(INTERIM / "llm_apply.jsonl")
    if llm is not None:
        # run_llm.py can select its subsample before method_labels exists; it must be a subset of it
        outside = int((~llm["entity_id"].isin(app_ids)).sum())
        n_mixed = int((~llm["entity_id"].isin(m["entity_id"])).sum()) - outside
        print(f"LLM subsample: {len(llm)} parsed entities, {outside} outside the C/D application sample, "
              f"{n_mixed} dropped as mixed-title clusters", flush=True)
        if outside:
            raise SystemExit("LLM subsample is not a subset of the application sample; rerun run_llm.py --mode apply")
        m = m.merge(llm, on="entity_id", how="left")
        has_e = m["E_roles"].notna()
        ef = role_flags(m.loc[has_e, "E_roles"], "E").rename(columns=lambda c: c[2:])
        flags["E"] = ef.reindex(m.index)          # NaN outside the LLM subsample
        for k in list(flags):
            flags[k] = flags[k].reindex(m.index)

    rows = []
    primary = [k for k in flags if "@" not in k]
    for meth, F in flags.items():
        for r in ROLE_COLS:
            col = F[r]
            ok = col.notna()
            d = m[ok].assign(_role=col[ok].astype(bool))
            yr = []
            for y, g in d.groupby("year"):
                gr = g[g["_role"]]
                yr.append({"year": int(y), "n": float(len(gr)), "k": float(gr["f"].sum()),
                           "share": float(gr["f"].mean()) if len(gr) else np.nan,
                           "prevalence": float(g["_role"].mean()), "n_pop": len(g)})
            yr = pd.DataFrame(yr)
            for _, z in yr.iterrows():
                rows.append({"kind": "yearly", "method": meth, "role": r, **z.to_dict()})
            t = hac_trend(yr)
            rows.append({"kind": "trend", "method": meth, "role": r, **t,
                         "mean_prevalence": float(d["_role"].mean()), "n_role": int(d["_role"].sum())})

    # H4: spread of female-share-in-role slopes across the primary methods (same population)
    tr = pd.DataFrame([x for x in rows if x["kind"] == "trend"])
    for r in ROLE_COLS:
        s = tr[(tr["role"] == r) & tr["method"].isin(["A", "B", "C", "D"])].dropna(subset=["slope_pp_dec"])
        if len(s) < 2:
            continue
        sl = s["slope_pp_dec"].to_numpy()
        signs = np.sign(sl)
        mn = np.min(np.abs(sl))
        rows.append({"kind": "h4", "role": r, "n_methods": len(sl), "slope_min": float(sl.min()),
                     "slope_max": float(sl.max()), "abs_ratio_max_min": float(np.max(np.abs(sl)) / mn) if mn > 0 else np.inf,
                     "signs_agree": bool((signs == signs[0]).all()),
                     "n_sig_positive": int(((s["ci_lo"] > 0)).sum()), "n_sig_negative": int(((s["ci_hi"] < 0)).sum()),
                     "methods": ",".join(s["method"])})

    # pairwise kappa by period and by year (time-dependent disagreement)
    for a, b in combinations(primary, 2):
        for r in ROLE_COLS:
            both = flags[a][r].notna() & flags[b][r].notna()
            if both.sum() < 50:
                continue
            xa, xb = flags[a].loc[both, r].astype(bool), flags[b].loc[both, r].astype(bool)
            per = m.loc[both, "period"]
            for p in ["all"] + list(per.cat.categories):
                sel = slice(None) if p == "all" else (per == p).to_numpy()
                ya, yb = xa[sel], xb[sel]
                if len(ya) < 30:
                    continue
                k = cohen_kappa_score(ya, yb) if (ya.nunique() > 1 or yb.nunique() > 1) else np.nan
                rows.append({"kind": "kappa", "method_a": a, "method_b": b, "role": r, "period": p, "kappa": k,
                             "agree": float((ya == yb).mean()), "prev_a": float(ya.mean()), "prev_b": float(yb.mean()),
                             "n": int(len(ya))})
            if a in ("A", "B", "C", "D") and b in ("A", "B", "C", "D"):
                ky = []
                for y in sorted(m.loc[both, "year"].unique()):
                    s_ = (m.loc[both, "year"] == y).to_numpy()
                    ya, yb = xa[s_], xb[s_]
                    ky.append({"year": int(y), "share": cohen_kappa_score(ya, yb) if (ya.any() or yb.any()) else np.nan,
                               "n": float(len(ya))})
                t = hac_trend(pd.DataFrame(ky))
                rows.append({"kind": "kappa_trend", "method_a": a, "method_b": b, "role": r,
                             "kappa_slope_per_decade": t["slope_pp_dec"] / 100, "ci_lo": t["ci_lo"] / 100,
                             "ci_hi": t["ci_hi"] / 100, "p": t["p"]})

    out = pd.DataFrame(rows)
    safe_write(lambda t: out.to_parquet(t, index=False), RESULTS / "method_agreement.parquet")

    # disagreement examples (public-domain excerpts, <= 420 chars)
    ctx = pd.concat([pd.read_parquet(p, columns=["entity_id", "ctx", "ctx_hl_start", "ctx_hl_end", "name"])
                     for p in sorted((INTERIM / "extract").glob("entities_*.parquet"))])
    ctx = ctx[ctx["entity_id"].isin(m["entity_id"])].set_index("entity_id")
    rng = np.random.default_rng(3)
    ex = []
    for r in ["AUTHORITY", "PUBLIC_OFFICE", "PROFESSIONAL", "BUSINESS", "CIVIC", "SOCIAL", "FAMILY", "CRIME_ACCIDENT",
              "ARTS_SPORTS", "LABOR", "MILITARY"]:
        lab = pd.DataFrame({k: flags[k][r] for k in primary})
        dis = lab.dropna().astype(bool)
        dis = dis[dis.nunique(axis=1) > 1]
        if dis.empty:
            continue
        pick = dis.index[rng.choice(len(dis), size=min(14, len(dis)), replace=False)]
        for i in pick:
            row = m.loc[i]
            c = ctx.loc[row["entity_id"]]
            s0, e0 = int(c["ctx_hl_start"]), int(c["ctx_hl_end"])
            a0 = max(0, s0 - 190)
            b0 = min(len(c["ctx"]), e0 + 190)
            ex.append({"role": r, "entity_id": row["entity_id"], "year": int(row["year"]),
                       "publication": row["publication"], "gender_signal": row["gender_hp"],
                       "excerpt": ("…" if a0 > 0 else "") + c["ctx"][a0:b0] + ("…" if b0 < len(c["ctx"]) else ""),
                       "hl_start": s0 - a0 + (1 if a0 > 0 else 0), "hl_end": e0 - a0 + (1 if a0 > 0 else 0),
                       "labels": {k: bool(dis.loc[i, k]) for k in dis.columns},
                       "probs": {pre: round(float(row[f"{pre}_{r}"]), 3) for pre in ("C", "D") if r != "AUTHORITY"}})
    with open(RESULTS / "disagreement_examples.json", "w") as fh:
        json.dump({"note": "Cases where role methods disagree. Excerpts are public-domain newspaper OCR (1900-1963), "
                           "unedited; they may contain offensive historical language.", "examples": ex}, fh,
                  ensure_ascii=False, indent=1)

    with pd.option_context("display.width", 220):
        print(out[out["kind"] == "h4"].round(2).to_string())
        k = out[(out["kind"] == "kappa") & (out["period"] == "all") & (out["role"] == "AUTHORITY")]
        print(k[["method_a", "method_b", "kappa", "agree", "prev_a", "prev_b", "n"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
