#!/usr/bin/env python
"""make analyze (step 7) -- evaluate H1-H7 by the rules fixed in research/hypotheses.md.

Status rules (pre-declared):
  supported    the pre-registered criterion holds in the primary specification AND the
               sign survives every cell of the robustness grid where one exists
  tentative    the criterion holds in the primary specification only
  unsupported  the criterion fails in the primary specification

Writes results/claims.json and research/claims_registry.md.  No status is set by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.models import hac_trend  # noqa: E402
from gna.paths import RESEARCH, RESULTS, safe_write  # noqa: E402

HEADLINE_ROLES = ["AUTHORITY", "PUBLIC_OFFICE", "PROFESSIONAL", "BUSINESS", "CIVIC"]


def f(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def main() -> int:
    ts = pd.read_parquet(RESULTS / "trend_summary.parquet")
    rt = pd.read_parquet(RESULTS / "role_trends.parquet")
    rb = pd.read_parquet(RESULTS / "robustness.parquet")
    ta = pd.read_parquet(RESULTS / "topic_adjusted.parquet")
    ma = pd.read_parquet(RESULTS / "method_agreement.parquet")
    ex = pd.read_parquet(RESULTS / "extraction_summary.parquet")
    qt = pd.read_parquet(RESULTS / "quote_trends.parquet")
    summ = rb[rb["kind"] == "summary"]
    claims = []

    def robust(estimand, method=None):
        s = summ[(summ["estimand"] == estimand) & (summ["method"].isna() if method is None else summ["method"] == method)]
        s = s[s.filter(["adjust", "tier", "papers"]).isna().all(axis=1)] if len(s) else s
        return None if s.empty else s.iloc[0]

    # ---------------------------------------------------------------- H1
    p = ts[(ts["estimand"] == "female_share") & (ts["tier"] == "hp") & (ts["population"] == "ner")].iloc[0]
    r = robust("H1_female_share")
    holds = p["ci_lo"] > 0
    status = ("supported" if holds and r is not None and bool(r["sign_survives_all"]) and r["slope_min"] > 0
              else "tentative" if holds else "unsupported")
    claims.append({"id": "H1", "claim": "The share of gender-signalled people who are signalled as women increases, 1900-1963.",
                   "status": status,
                   "evidence": {"slope_pp_per_decade": f(p["slope_pp_dec"]), "ci": [f(p["ci_lo"]), f(p["ci_hi"])],
                                "early_share": f(p["early_share"], 4), "late_share": f(p["late_share"], 4),
                                "multiverse_share_positive": f(r["share_positive"], 3) if r is not None else None,
                                "multiverse_cells": int(r["n_cells"]) if r is not None else None,
                                "multiverse_slope_range": [f(r["slope_min"]), f(r["slope_max"])] if r is not None else None},
                   "script": "scripts/analyze_trends.py, scripts/robustness.py", "result": "results/trend_summary.parquet",
                   "figure": "figures/fig03_female_share.png", "page": "/timeline/"})

    # ---------------------------------------------------------------- H2
    per_method = {}
    for m in ("A", "B"):
        allp = rt[(rt["estimand"] == "female_share") & (rt["tier"] == "hp") & (rt["population"] == "ner")].set_index("year")
        au = rt[(rt["estimand"] == "female_share_in_role") & (rt["role"] == "AUTHORITY") & (rt["method"] == m)].set_index("year")
        j = allp[["share", "n"]].join(au[["share", "n"]], rsuffix="_auth", how="inner").dropna()
        lower_every_year = bool((j["share_auth"] < j["share"]).all())
        gap = pd.DataFrame({"year": j.index, "share": j["share_auth"] - j["share"], "n": j["n_auth"]})
        t = hac_trend(gap)
        per_method[m] = {"lower_in_every_year": lower_every_year, "gap_trend_pp_per_decade": f(t["slope_pp_dec"]),
                         "gap_trend_ci": [f(t["ci_lo"]), f(t["ci_hi"])], "gap_narrows": bool(t["ci_lo"] > 0)}
    mt = ma[(ma["kind"] == "trend") & (ma["role"] == "AUTHORITY")]
    ok_all = all(v["lower_in_every_year"] and v["gap_narrows"] for v in per_method.values())
    ok_any = any(v["lower_in_every_year"] and v["gap_narrows"] for v in per_method.values())
    r = robust("H2_female_share_in_AUTHORITY", "B")
    status = ("supported" if ok_all and r is not None and bool(r["sign_survives_all"]) else
              "tentative" if ok_any else "unsupported")
    claims.append({"id": "H2", "claim": "Women's share of authority roles is below their share of all people in every period, and the gap narrows.",
                   "status": status, "evidence": {"by_method": per_method,
                   "authority_slope_by_method_same_entities": {row["method"]: [f(row["slope_pp_dec"]), f(row["ci_lo"]), f(row["ci_hi"])]
                                                               for _, row in mt.iterrows()},
                   "multiverse_B_share_positive": f(r["share_positive"], 3) if r is not None else None},
                   "script": "scripts/analyze_trends.py, scripts/analyze_methods.py", "result": "results/role_trends.parquet",
                   "figure": "figures/fig04_roles_female_share.png", "page": "/roles/"})

    # ---------------------------------------------------------------- H3
    lp = ta[(ta["kind"] == "lpm") & (ta["outcome"] == "female_share") & (ta["sample"] == "all")].set_index("fe")
    raw, adj = lp.loc["none"], lp.loc["topic+paper"]
    rel = abs(adj["slope_pp_dec"] - raw["slope_pp_dec"]) / abs(raw["slope_pp_dec"]) if raw["slope_pp_dec"] else np.nan
    dc = ta[(ta["kind"] == "decomposition") & (ta["outcome"] == "female_share")]
    status = "supported" if rel >= 0.25 else "unsupported"
    claims.append({"id": "H3", "claim": "Adjusting for topic and newspaper changes the 1900-1963 trend by at least 25% of the raw slope.",
                   "status": status, "evidence": {"raw_slope": f(raw["slope_pp_dec"]), "raw_ci": [f(raw["ci_lo"]), f(raw["ci_hi"])],
                   "adjusted_slope": f(adj["slope_pp_dec"]), "adjusted_ci": [f(adj["ci_lo"]), f(adj["ci_hi"])],
                   "relative_change": f(rel, 3),
                   "decomposition": {row["group"]: {"total_pp": f(row["total_pp"]), "composition_pp": f(row["composition_pp"]),
                                                    "within_pp": f(row["within_pp"])} for _, row in dc.iterrows()}},
                   "script": "scripts/analyze_composition.py", "result": "results/topic_adjusted.parquet",
                   "figure": "figures/fig06_raw_vs_adjusted.png", "page": "/topics/"})

    # ---------------------------------------------------------------- H4
    h4 = ma[(ma["kind"] == "h4") & ma["role"].isin(HEADLINE_ROLES)]
    hits = h4[(h4["abs_ratio_max_min"] > 2) | (~h4["signs_agree"].astype(bool))]
    claims.append({"id": "H4", "claim": "Role-measurement methods produce materially different trend effect sizes (ratio > 2 or sign disagreement for at least one headline role).",
                   "status": "supported" if len(hits) else "unsupported",
                   "evidence": {row["role"]: {"slope_min": f(row["slope_min"]), "slope_max": f(row["slope_max"]),
                                              "ratio": f(row["abs_ratio_max_min"]), "signs_agree": bool(row["signs_agree"]),
                                              "methods": row["methods"]} for _, row in h4.iterrows()},
                   "script": "scripts/analyze_methods.py", "result": "results/method_agreement.parquet",
                   "figure": "figures/fig08_method_comparison.png", "page": "/methods/"})

    # ---------------------------------------------------------------- H5
    tr = ma[(ma["kind"] == "trend") & ma["role"].isin(HEADLINE_ROLES)].pivot_table(index="role", columns="method", values="slope_pp_dec")
    per_role = {}
    for role, row in tr.iterrows():
        if {"A", "B", "D"} <= set(row.index) and row[["A", "B", "D"]].notna().all():
            per_role[role] = {"A": f(row["A"]), "B": f(row["B"]), "D": f(row["D"]),
                              "lexical_largest": bool(abs(row["A"]) > abs(row["B"]) and abs(row["A"]) > abs(row["D"]))}
    n_true = sum(v["lexical_largest"] for v in per_role.values())
    claims.append({"id": "H5", "claim": "The lexical method (A) yields a larger absolute trend than the context-aware methods (B, D).",
                   "status": "supported" if per_role and n_true == len(per_role) else ("tentative" if n_true > len(per_role) / 2 else "unsupported"),
                   "evidence": {"roles": per_role, "n_roles_lexical_largest": n_true, "n_roles": len(per_role)},
                   "script": "scripts/analyze_methods.py", "result": "results/method_agreement.parquet",
                   "figure": "figures/fig08_method_comparison.png", "page": "/methods/"})

    # ---------------------------------------------------------------- H6
    u = hac_trend(pd.DataFrame({"year": ex["year"], "share": ex["unknown_share_hp"], "n": ex["n_entities_ner"].astype(float)}))
    slopes = {}
    for col in ("female_share_classified", "female_share_lower_bound", "female_share_upper_bound"):
        t = hac_trend(pd.DataFrame({"year": ex["year"], "share": ex[col], "n": ex["n_entities_ner"].astype(float)}))
        slopes[col] = [f(t["slope_pp_dec"]), f(t["ci_lo"]), f(t["ci_hi"])]
    changes = u["ci_lo"] > 0 or u["ci_hi"] < 0
    signs = {np.sign(v[0]) for v in slopes.values() if v[0] is not None}
    sensitive = len(signs) > 1
    claims.append({"id": "H6", "claim": "The UNKNOWN share changes over time, and the women's-share trend is sensitive to how UNKNOWN is handled.",
                   "status": "supported" if changes and sensitive else ("tentative" if changes or sensitive else "unsupported"),
                   "evidence": {"unknown_slope_pp_per_decade": f(u["slope_pp_dec"]), "unknown_ci": [f(u["ci_lo"]), f(u["ci_hi"])],
                                "unknown_share_first": f(ex["unknown_share_hp"].iloc[0], 4), "unknown_share_last": f(ex["unknown_share_hp"].iloc[-1], 4),
                                "slopes_by_handling": slopes, "signs_differ_across_handling": sensitive},
                   "script": "scripts/analyze_trends.py", "result": "results/extraction_summary.parquet",
                   "figure": "figures/fig02_unknown_share.png", "page": "/timeline/"})

    # ---------------------------------------------------------------- H7
    h7 = qt[(qt["estimand"] == "H7_speaker_minus_mention_share") & (qt["attribution"] == "name")]
    pooled = h7[h7["period"] == "all"].iloc[0]
    pers = h7[h7["period"] != "all"]
    all_neg = bool((pers["ci_hi"] < 0).all())
    claims.append({"id": "H7", "claim": "Among attributed quotations, women's share of speakers is lower than their share of gender-signalled people.",
                   "status": "supported" if pooled["ci_hi"] < 0 and all_neg else ("tentative" if pooled["ci_hi"] < 0 else "unsupported"),
                   "evidence": {"pooled_diff_pp": f(pooled["diff_pp"]), "pooled_ci": [f(pooled["ci_lo"]), f(pooled["ci_hi"])],
                                "by_period": {row["period"]: [f(row["diff_pp"]), f(row["ci_lo"]), f(row["ci_hi"])] for _, row in pers.iterrows()}},
                   "script": "scripts/analyze_trends.py", "result": "results/quote_trends.parquet",
                   "figure": "figures/fig12_quotes.png", "page": "/quotes/"})

    out = {"rules": __doc__.split("Status rules (pre-declared):")[1].split("Writes")[0].strip(), "claims": claims}
    safe_write(lambda t: Path(t).write_text(json.dumps(out, indent=1)), RESULTS / "claims.json")

    lines = ["# Claims registry", "", "_Generated by `scripts/claims.py` from frozen results. Do not edit by hand._", "",
             "Status rules (fixed in `research/hypotheses.md` before outcomes were computed):", "",
             *[f"- {ln.strip()}" for ln in out["rules"].splitlines() if ln.strip() and not ln.startswith(" " * 15)], "",
             "| id | claim | status | analysis script | result file | figure |", "|---|---|---|---|---|---|"]
    for c in claims:
        lines.append(f"| {c['id']} | {c['claim']} | **{c['status']}** | `{c['script']}` | `{c['result']}` | `{c['figure']}` |")
    lines += ["", "## Evidence", ""]
    for c in claims:
        lines += [f"### {c['id']} — {c['status']}", "", c["claim"], "", "```json", json.dumps(c["evidence"], indent=1), "```", ""]
    (RESEARCH / "claims_registry.md").write_text("\n".join(lines))
    for c in claims:
        print(f"{c['id']}: {c['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
