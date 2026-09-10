#!/usr/bin/env python
"""make analyze (step 2) -- longitudinal estimates for H1, H2, H6, H7 plus diagnostics.

Outputs
  results/role_trends.parquet          yearly series (share + cluster-bootstrap CI)
  results/trend_summary.parquet        HAC slopes (pp/decade) and period contrasts per series
  results/quote_trends.parquet         speaker vs mention shares, quote rates, H7 contrasts
  results/extraction_summary.parquet   per-year extraction diagnostics incl. UNKNOWN share
  results/gender_consistency.parquet   honorific vs pronoun-rule agreement (tier-2 accuracy proxy)
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.frame import ROLE_COLS, load  # noqa: E402
from gna.models import hac_trend, unknown_bounds, yearly_share  # noqa: E402
from gna.paths import INTERIM, RESULTS, safe_write  # noqa: E402

B = 400
EARLY, LATE = (1900, 1921), (1948, 1963)


def series(df, success, **tags) -> pd.DataFrame:
    d = df.dropna(subset=[success])
    if d.empty:
        return pd.DataFrame()
    s = yearly_share(d, success, B=B, seed=zlib.crc32(repr(sorted(tags.items())).encode()))
    for k, v in tags.items():
        s[k] = v
    return s


def summarize(s: pd.DataFrame, keys: list[str]) -> list[dict]:
    out = []
    for key, g in s.groupby(keys, dropna=False, sort=False):
        t = hac_trend(g)
        e = g[(g["year"] >= EARLY[0]) & (g["year"] <= EARLY[1])]
        l_ = g[(g["year"] >= LATE[0]) & (g["year"] <= LATE[1])]
        pe = e["k"].sum() / e["n"].sum() if e["n"].sum() else np.nan
        pl = l_["k"].sum() / l_["n"].sum() if l_["n"].sum() else np.nan
        out.append({**dict(zip(keys, key if isinstance(key, tuple) else (key,))), **t,
                    "early_share": pe, "late_share": pl, "early_late_diff_pp": 100 * (pl - pe),
                    "n_total": float(g["n"].sum()), "min_year_n": float(g["n"].min())})
    return out


def main() -> int:
    df = load()
    base = df[df["ner_ok"]].copy()
    rows = []

    # ---- H1: female share among gender-signalled entities, by tier
    for tier in ("h", "hp", "hpn"):
        rows.append(series(base, f"f_{tier}", estimand="female_share", role="ALL", method="-", tier=tier,
                           population="ner"))
    rows.append(series(df, "f_hp", estimand="female_share", role="ALL", method="-", tier="hp",
                       population="ner+pattern"))

    # ---- H2 and role composition: methods A and B on the full population
    for m in ("A", "B"):
        for r in ROLE_COLS:
            sub = base[base[f"{m}_{r}"]]
            rows.append(series(sub, "f_hp", estimand="female_share_in_role", role=r, method=m, tier="hp",
                               population="ner"))
        for g, gname in ((1.0, "F"), (0.0, "M")):
            gd = base[base["f_hp"] == g]
            for r in ROLE_COLS:
                gd = gd.assign(_y=gd[f"{m}_{r}"].astype(float))
                rows.append(series(gd, "_y", estimand=f"role_rate_{gname}", role=r, method=m, tier="hp",
                                   population="ner"))

    yearly = pd.concat([r for r in rows if len(r)], ignore_index=True)
    safe_write(lambda t: yearly.to_parquet(t, index=False), RESULTS / "role_trends.parquet")
    summ = pd.DataFrame(summarize(yearly, ["estimand", "role", "method", "tier", "population"]))
    # role-rate gap F - M (pp) trend, from the two rate series
    gaps = []
    for (m, r), g in yearly[yearly["estimand"].str.startswith("role_rate")].groupby(["method", "role"]):
        piv = g.pivot_table(index="year", columns="estimand", values=["share", "n"])
        if ("share", "role_rate_F") not in piv or ("share", "role_rate_M") not in piv:
            continue
        gd = pd.DataFrame({"year": piv.index, "share": piv[("share", "role_rate_F")] - piv[("share", "role_rate_M")],
                           "n": np.minimum(piv[("n", "role_rate_F")], piv[("n", "role_rate_M")])})
        gaps.append({"estimand": "role_rate_gap_F_minus_M", "role": r, "method": m, "tier": "hp",
                     "population": "ner", **hac_trend(gd)})
    summ = pd.concat([summ, pd.DataFrame(gaps)], ignore_index=True)
    safe_write(lambda t: summ.to_parquet(t, index=False), RESULTS / "trend_summary.parquet")

    # ---- H6: classifiability over time + UNKNOWN bounds
    ex = []
    arts = pd.read_parquet(INTERIM / "articles_parse.parquet")
    for y, g in base.groupby("year"):
        a = arts[arts["year"] == y]
        n_f, n_m = float((g["f_hp"] == 1).sum()), float((g["f_hp"] == 0).sum())
        n_u = float(g["f_hp"].isna().sum())
        b = unknown_bounds(n_f, n_m, n_u)
        ex.append({"year": int(y), "n_articles": len(a), "n_entities_ner": len(g),
                   "entities_per_article": len(g) / max(len(a), 1),
                   "unknown_share_hp": n_u / len(g), "unknown_share_h": float(g["f_h"].isna().mean()),
                   "unknown_share_hpn": float(g["f_hpn"].isna().mean()),
                   "share_src_honorific": float((g["src_hp"] == "honorific").mean()),
                   "share_src_pronoun": float((g["src_hp"] == "pronoun").mean()),
                   "share_src_conflict": float((g["src_hp"] == "conflict").mean()),
                   "pattern_only_entities": int(((df["year"] == y) & ~df["ner_ok"]).sum()),
                   "ambiguous_dropped_per_article": float(a["n_ambiguous_dropped"].mean()),
                   "mentions_per_entity": float(g["n_mentions"].mean()),
                   "female_share_classified": b["share_f_classified"],
                   "female_share_lower_bound": b["lower"], "female_share_upper_bound": b["upper"]})
    exs = pd.DataFrame(ex)
    safe_write(lambda t: exs.to_parquet(t, index=False), RESULTS / "extraction_summary.parquet")

    # ---- tier-2 accuracy proxy: pronoun rule vs honorific where both exist
    both = base[base["hclass"].isin(["F", "M"]) & base["pron_signal"].isin(["F", "M"])]
    gc = both.groupby("year").apply(lambda g: pd.Series({
        "n": len(g), "agree": float((g["hclass"] == g["pron_signal"]).mean()),
        "agree_when_F": float((g.loc[g["hclass"] == "F", "pron_signal"] == "F").mean()),
        "agree_when_M": float((g.loc[g["hclass"] == "M", "pron_signal"] == "M").mean())}), include_groups=False
    ).reset_index()
    tot = pd.DataFrame([{"year": -1, "n": len(both), "agree": float((both["hclass"] == both["pron_signal"]).mean()),
                         "agree_when_F": float((both.loc[both["hclass"] == "F", "pron_signal"] == "F").mean()),
                         "agree_when_M": float((both.loc[both["hclass"] == "M", "pron_signal"] == "M").mean())}])
    gcs = pd.concat([gc, tot], ignore_index=True)
    safe_write(lambda t: gcs.to_parquet(t, index=False), RESULTS / "gender_consistency.parquet")

    # ---- H7: quotation
    q = []
    gs = base.dropna(subset=["f_hp"])
    for col, label in (("quoted_name", "name"), ("quoted_any", "name+pronoun")):
        spk = gs[gs[col]]
        q.append(series(spk, "f_hp", estimand="female_share_speakers", attribution=label))
        q.append(series(gs.assign(_y=gs[col].astype(float))[gs["f_hp"] == 1], "_y", estimand="quote_rate_F",
                        attribution=label))
        q.append(series(gs.assign(_y=gs[col].astype(float))[gs["f_hp"] == 0], "_y", estimand="quote_rate_M",
                        attribution=label))
    q.append(series(gs, "f_hp", estimand="female_share_mentions", attribution="-"))
    dq = gs[gs["quoted_name"]].assign(_y=lambda d: (d["n_direct_name"] > 0).astype(float))
    for g, name in ((1.0, "F"), (0.0, "M")):
        q.append(series(dq[dq["f_hp"] == g], "_y", estimand=f"direct_share_{name}", attribution="name"))
    qt = pd.concat([x for x in q if len(x)], ignore_index=True)

    # pooled H7 contrast with an article-cluster bootstrap: speaker share minus mention share
    rng = np.random.default_rng(7)
    contr = []
    for label, col in (("name", "quoted_name"), ("name+pronoun", "quoted_any")):
        for per, g in [("all", gs)] + [(str(p), gg) for p, gg in gs.groupby("period", observed=True)]:
            g = g.assign(_fq=g["f_hp"] * g[col].astype(float), _q=g[col].astype(float))
            agg = g.groupby("article_id").agg(nf=("f_hp", "sum"), n=("f_hp", "size"), sf=("_fq", "sum"),
                                              sn=("_q", "sum"))
            nf, n, sf, sn = (agg[c].to_numpy(float) for c in ("nf", "n", "sf", "sn"))
            pw = rng.poisson(1.0, size=(300, len(agg)))
            d = (pw @ sf) / np.maximum(pw @ sn, 1) - (pw @ nf) / np.maximum(pw @ n, 1)
            est = sf.sum() / max(sn.sum(), 1) - nf.sum() / n.sum()
            contr.append({"estimand": "H7_speaker_minus_mention_share", "attribution": label, "period": per,
                          "diff_pp": 100 * est, "ci_lo": 100 * np.quantile(d, 0.025), "ci_hi": 100 * np.quantile(d, 0.975),
                          "n_speakers": int(sn.sum()), "n_mentions": int(n.sum())})
    qt = pd.concat([qt, pd.DataFrame(contr)], ignore_index=True)
    safe_write(lambda t: qt.to_parquet(t, index=False), RESULTS / "quote_trends.parquet")

    with pd.option_context("display.width", 220):
        print(exs[["year", "entities_per_article", "unknown_share_hp", "share_src_honorific", "share_src_pronoun",
                   "female_share_classified", "female_share_lower_bound", "female_share_upper_bound"]].round(3).to_string())
        print(gcs.round(3).tail(3).to_string())
        show = summ[(summ["estimand"].isin(["female_share", "female_share_in_role"])) &
                    (summ["role"].isin(["ALL", "AUTHORITY", "PUBLIC_OFFICE", "PROFESSIONAL", "BUSINESS", "CIVIC",
                                        "SOCIAL", "FAMILY"]))]
        print(show[["estimand", "role", "method", "tier", "population", "slope_pp_dec", "ci_lo", "ci_hi",
                    "early_share", "late_share"]].round(3).to_string())
        print(pd.DataFrame(contr).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
