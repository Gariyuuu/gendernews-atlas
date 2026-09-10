#!/usr/bin/env python
"""make analyze (step 3) -- composition: FE-adjusted trends and shift-share decompositions.

Outputs results/topic_adjusted.parquet with row kinds
  lpm            trend (pp/decade) for an outcome under a fixed-effects spec and sample
  decomposition  early->late change split into composition vs within-group components
  group_period   female share by topic/section/newspaper-group x period (for the site)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.frame import load  # noqa: E402
from gna.models import lpm_trend  # noqa: E402
from gna.paths import RESULTS, safe_write  # noqa: E402

EARLY, LATE = (1900, 1921), (1948, 1963)
FE_SPECS = {"none": [], "topic": ["topic_label"], "paper": ["publication"],
            "topic+paper": ["topic_label", "publication"],
            "topic+paper+page": ["topic_label", "publication", "page_band"]}


def samples(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"all": d, "excl_evening_star": d[~d["evening_star"]], "news_only": d[d["is_news"]],
            "pre_1922": d[d["year"] <= 1921], "post_1923": d[d["year"] >= 1924]}


def decompose(d: pd.DataFrame, group: str, y: str = "f_hp", B: int = 300, seed: int = 0) -> dict:
    """Symmetric Kitagawa shift-share of the EARLY->LATE change, article-cluster bootstrap CI."""
    d = d[(d["year"].between(*EARLY)) | (d["year"].between(*LATE))].dropna(subset=[y, group])
    d = d.assign(_late=d["year"] >= LATE[0])
    art = d.groupby("article_id").agg(late=("_late", "first"), grp=(group, "first"), n=(y, "size"), k=(y, "sum"))
    gcode, glev = pd.factorize(art["grp"])
    G = len(glev)
    late = art["late"].to_numpy()
    n, k = art["n"].to_numpy(float), art["k"].to_numpy(float)

    def parts(wt):
        out = {}
        for p, mask in (("E", ~late), ("L", late)):
            nn = np.bincount(gcode[mask], weights=(wt * n)[mask], minlength=G)
            kk = np.bincount(gcode[mask], weights=(wt * k)[mask], minlength=G)
            out[p] = (nn / nn.sum(), np.divide(kk, nn, out=np.full(G, np.nan), where=nn > 0))
        (wE, sE), (wL, sL) = out["E"], out["L"]
        ok = ~np.isnan(sE) & ~np.isnan(sL)              # groups present in both periods
        comp = float(np.sum((wL - wE)[ok] * (sL + sE)[ok] / 2))
        within = float(np.sum((sL - sE)[ok] * (wL + wE)[ok] / 2))
        total = float(np.nansum(wL * sL) - np.nansum(wE * sE))
        return total, comp, within, float(1 - (wE[ok].sum() + wL[ok].sum()) / 2)

    tot, comp, within, unmatched = parts(np.ones(len(art)))
    rng = np.random.default_rng(seed)
    bs = np.array([parts(rng.poisson(1.0, len(art)))[:3] for _ in range(B)])
    lo, hi = np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)
    return {"kind": "decomposition", "group": group, "total_pp": 100 * tot, "total_lo": 100 * lo[0], "total_hi": 100 * hi[0],
            "composition_pp": 100 * comp, "composition_lo": 100 * lo[1], "composition_hi": 100 * hi[1],
            "within_pp": 100 * within, "within_lo": 100 * lo[2], "within_hi": 100 * hi[2],
            "share_in_unmatched_groups": unmatched, "n_groups": G}


def adjusted_yearly(d: pd.DataFrame, y: str = "f_hp", fe: tuple = ("topic_label", "publication"),
                    B: int = 120, seed: int = 0) -> pd.DataFrame:
    """Composition-adjusted yearly shares: y = a_year + g_topic + d_paper (LPM, alternating
    projections); adjusted share_t = a_t + pooled mean of (g + d).  Article-cluster
    Poisson bootstrap for 95% CIs.  Reads as 'the share each year would have had with the
    pooled 1900-63 topic and newspaper mix'."""
    d = d.dropna(subset=[y, *fe])
    yr = pd.factorize(d["year"], sort=True)
    codes = [yr[0]] + [pd.factorize(d[f])[0] for f in fe]
    yy = d[y].to_numpy(float)
    art = pd.factorize(d["article_id"])[0]

    def fit(w):
        effs = [np.zeros(c.max() + 1) for c in codes]
        for _ in range(60):
            delta = 0.0
            for j, c in enumerate(codes):
                other = sum(effs[k][codes[k]] for k in range(len(codes)) if k != j)
                num = np.bincount(c, weights=w * (yy - other), minlength=len(effs[j]))
                den = np.bincount(c, weights=w, minlength=len(effs[j]))
                new = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                if j > 0:
                    new -= np.average(new[c], weights=w)          # identify level on the year effects
                delta = max(delta, float(np.abs(new - effs[j]).max()))
                effs[j] = new
            if delta < 1e-7:
                break
        comp = np.average(sum(effs[k][codes[k]] for k in range(1, len(codes))), weights=w)
        return effs[0] + comp

    base = fit(np.ones(len(yy)))
    rng = np.random.default_rng(seed)
    n_art = art.max() + 1
    boots = np.array([fit(rng.poisson(1.0, n_art)[art].astype(float)) for _ in range(B)])
    lo, hi = np.nanquantile(boots, [0.025, 0.975], axis=0)
    n = np.bincount(yr[0])
    return pd.DataFrame({"year": yr[1].astype(int), "share": base, "ci_lo": lo, "ci_hi": hi, "n": n.astype(float)})


def main() -> int:
    df = load()
    base = df[df["ner_ok"] & df["f_hp"].notna()].copy()
    base["paper_group"] = np.where(base["evening_star"], "evening star", "other papers")
    outcomes = {
        "female_share": base,
        "female_share_in_AUTHORITY_B": base[base["B_AUTHORITY"]],
        "female_share_in_AUTHORITY_A": base[base["A_AUTHORITY"]],
        "female_share_in_CIVIC_B": base[base["B_CIVIC"]],
    }
    rows = []
    for oname, od in outcomes.items():
        for sname, sd in samples(od).items():
            for fname, fe in FE_SPECS.items():
                if sname in ("pre_1922", "post_1923") and fname not in ("none", "topic+paper"):
                    continue
                r = lpm_trend(sd, "f_hp", fe=fe, cluster="publication", cluster2="year")
                rows.append({"kind": "lpm", "outcome": oname, "sample": sname, "fe": fname, **r})
                print(f"{oname:30s} {sname:18s} {fname:18s} slope={r['slope_pp_dec']:+.2f} "
                      f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] n={r['n']}", flush=True)

    for oname in ("female_share", "female_share_in_AUTHORITY_B"):
        for grp in ("topic_label", "section", "paper_group"):
            d = decompose(outcomes[oname], grp)
            rows.append({**d, "outcome": oname})
            print(f"DECOMP {oname} by {grp}: total={d['total_pp']:+.2f} comp={d['composition_pp']:+.2f} "
                  f"within={d['within_pp']:+.2f} unmatched={d['share_in_unmatched_groups']:.3f}", flush=True)

    for oname in ("female_share", "female_share_in_AUTHORITY_B", "female_share_in_CIVIC_B"):
        adj = adjusted_yearly(outcomes[oname])
        for _, r in adj.iterrows():
            rows.append({"kind": "adjusted_yearly", "outcome": oname, "fe": "topic+paper", **r.to_dict()})
        print(f"ADJUSTED {oname}: " + " ".join(f"{int(a)}={b:.3f}" for a, b in zip(adj['year'], adj['share'])), flush=True)

    for grp in ("topic_label", "section", "paper_group"):
        t = base.groupby([grp, "period"], observed=True)["f_hp"].agg(["mean", "size"]).reset_index()
        tot = base.groupby("period", observed=True).size()
        for _, r in t.iterrows():
            rows.append({"kind": "group_period", "group": grp, "group_value": r[grp], "period": str(r["period"]),
                         "female_share": r["mean"], "n": int(r["size"]),
                         "entity_share_of_period": r["size"] / tot[r["period"]]})

    out = pd.DataFrame(rows)
    safe_write(lambda t: out.to_parquet(t, index=False), RESULTS / "topic_adjusted.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
