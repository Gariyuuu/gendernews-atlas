#!/usr/bin/env python
"""make robustness -- multiverse over analysis choices for the headline estimands.

A headline claim is kept only if its sign survives every cell of its grid
(research/hypotheses.md, "Decisions fixed in advance").  Cells use the fast aggregate
estimator (yearly/binned shares -> HAC trend); FE-adjusted variants use the LPM on a
reduced grid.

Output results/robustness.parquet with kinds: cell, summary.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.frame import load  # noqa: E402
from gna.lexicons import AUTHORITY, ROLES  # noqa: E402
from gna.models import hac_trend, lpm_trend  # noqa: E402
from gna.paths import INTERIM, RESULTS, safe_write  # noqa: E402

GRID = {
    "tier": ["h", "hp", "hpn"],
    "population": ["ner", "ner+pattern"],
    "dedup": ["keep", "drop_near"],
    "ocr": ["all", "clean"],
    "papers": ["all", "excl_evening_star", "capped"],
    "bins": ["year", "bin5", "decade"],
    "content": ["all", "news_only"],
    "resolution": ["mixed_unknown", "v2_merge"],   # decision D17: mixed-title clusters UNKNOWN vs kept
}


def select(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    if c.get("population", "ner") == "ner":
        m &= df["ner_ok"].to_numpy()
    if c.get("dedup") == "drop_near":
        m &= ~df["near_dup"].fillna(False).to_numpy(bool)
    if c.get("ocr") == "clean":
        m &= df["ocr_clean"].fillna(False).to_numpy(bool)
    if c.get("papers") == "excl_evening_star":
        m &= ~df["evening_star"].to_numpy(bool)
    elif c.get("papers") == "capped":
        m &= df["in_cap"].fillna(False).to_numpy(bool)
    if c.get("content") == "news_only":
        m &= df["is_news"].fillna(True).to_numpy(bool)
    return df[m]


def binned_trend(d: pd.DataFrame, y: str, bins: str) -> dict:
    d = d.dropna(subset=[y])
    key = {"year": "year", "bin5": "bin5", "decade": "decade"}[bins]
    g = d.groupby(key)[y].agg(["sum", "size"]).reset_index()
    g = g.rename(columns={key: "t"})
    if bins != "year":                                    # bin midpoint as the time value
        mid = d.groupby(key)["year"].mean()
        g["t"] = g["t"].map(mid)
    g = pd.DataFrame({"year": g["t"], "share": g["sum"] / g["size"], "n": g["size"].astype(float)})
    r = hac_trend(g, maxlags=2 if bins == "year" else 1)
    r["n"] = int(len(d))
    return r


def summarize(cells: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = []
    for k, g in cells.groupby(keys):
        s = g["slope_pp_dec"].dropna()
        out.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "kind": "summary", "n_cells": len(s),
                    "share_positive": float((s > 0).mean()), "share_sig_positive": float((g["ci_lo"] > 0).mean()),
                    "share_sig_negative": float((g["ci_hi"] < 0).mean()), "slope_min": float(s.min()),
                    "slope_median": float(s.median()), "slope_max": float(s.max()),
                    "sign_survives_all": bool((s > 0).all() or (s < 0).all())})
    return pd.DataFrame(out)


def main() -> int:
    df = load()
    cells = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        c = dict(zip(keys, combo))
        d = select(df, c)
        y = f"f_{c['tier']}" + ("_v2" if c["resolution"] == "v2_merge" else "")
        cells.append({"estimand": "H1_female_share", "method": "-", **c, **binned_trend(d, y, c["bins"])})
        for m in ("A", "B"):
            dd = d[d[f"{m}_AUTHORITY"]]
            cells.append({"estimand": "H2_female_share_in_AUTHORITY", "method": m, **c, **binned_trend(dd, y, c["bins"])})
            dd = d[d[f"{m}_CIVIC"]]
            cells.append({"estimand": "female_share_in_CIVIC", "method": m, **c, **binned_trend(dd, y, c["bins"])})
    print(f"aggregate cells: {len(cells)}", flush=True)

    # methods C/D (thresholds) and E on the shared application subsample
    ml = INTERIM / "method_labels.parquet"
    if ml.exists():
        m = pd.read_parquet(ml)
        m = m.merge(df[["entity_id", "ner_ok", "near_dup", "ocr_clean", "evening_star", "in_cap", "is_news", "bin5",
                        "decade", "f_h", "f_hp"]], on="entity_id", how="left")
        auth = [r for r in ROLES if r in AUTHORITY]
        sub_grid = {"tier": ["h", "hp"], "ocr": ["all", "clean"], "papers": ["all", "excl_evening_star", "capped"],
                    "bins": ["year", "bin5", "decade"], "dedup": ["keep", "drop_near"], "content": ["all", "news_only"]}
        for combo in itertools.product(*sub_grid.values()):
            c = dict(zip(sub_grid, combo))
            d = select(m.assign(population="ner"), {**c, "population": "ner"})
            y = f"f_{c['tier']}"
            for pre in ("C", "D"):
                for th in (0.3, 0.5, 0.7):
                    flag = (d[[f"{pre}_{r}" for r in auth]] >= th).any(axis=1)
                    cells.append({"estimand": "H2_female_share_in_AUTHORITY", "method": f"{pre}@{th}", "population": "app",
                                  **c, **binned_trend(d[flag], y, c["bins"])})
            for meth, col in (("A", "roles_a"), ("B", "roles_b")):
                flag = d[col].map(lambda rl: bool(set(rl) & AUTHORITY))
                cells.append({"estimand": "H2_female_share_in_AUTHORITY", "method": f"{meth}(app)", "population": "app",
                              **c, **binned_trend(d[flag], y, c["bins"])})
        print(f"cells incl. application subsample: {len(cells)}", flush=True)

    # FE-adjusted LPM on a reduced grid
    for tier, papers, ocr in itertools.product(["h", "hp", "hpn"], GRID["papers"], GRID["ocr"]):
        d = select(df, {"population": "ner", "papers": papers, "ocr": ocr})
        for est, dd in (("H1_female_share", d), ("H2_female_share_in_AUTHORITY", d[d["B_AUTHORITY"]])):
            for fe_name, fe in (("none", []), ("topic+paper", ["topic_label", "publication"])):
                r = lpm_trend(dd, f"f_{tier}", fe=fe)
                cells.append({"estimand": est, "method": "-" if est.startswith("H1") else "B", "tier": tier,
                              "population": "ner", "papers": papers, "ocr": ocr, "dedup": "keep", "bins": "micro",
                              "content": "all", "adjust": fe_name, **r})

    cell_df = pd.DataFrame(cells)
    cell_df["adjust"] = cell_df["adjust"].fillna("raw")
    cell_df["resolution"] = cell_df["resolution"].fillna("mixed_unknown")
    cell_df["kind"] = "cell"
    summ = pd.concat([
        summarize(cell_df, ["estimand"]),
        # like-for-like: only the aggregate grid is run under both resolutions (FE and app cells are primary-only)
        summarize(cell_df[cell_df["population"].isin(["ner", "ner+pattern"]) & (cell_df["adjust"] == "raw")],
                  ["estimand", "resolution"]),
        summarize(cell_df, ["estimand", "method"]),
        summarize(cell_df, ["estimand", "adjust"]),
        summarize(cell_df, ["estimand", "tier"]),
        summarize(cell_df, ["estimand", "papers"]),
    ], ignore_index=True)
    out = pd.concat([cell_df, summ], ignore_index=True)
    safe_write(lambda t: out.to_parquet(t, index=False), RESULTS / "robustness.parquet")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
