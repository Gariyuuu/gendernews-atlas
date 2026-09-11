#!/usr/bin/env python
"""Fixes requested in the mock reviews (research/reviews.md), implemented as checks.

1. Few clusters: wild-cluster restricted bootstrap (WCR, Rademacher weights, clusters =
   years, 22 of them) p-values for the headline LPM slopes, raw and FE-adjusted.
2. Functional form: logit average marginal effect of decade vs the LPM slope, raw and with
   topic fixed effects.
3. Multiple testing: Benjamini-Hochberg q-values across the family of role-level trend tests.

Output: results/reviewer_checks.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from gna.frame import load  # noqa: E402
from gna.models import _demean, bh_fdr, dec, lpm_trend  # noqa: E402
from gna.paths import RESULTS, safe_write  # noqa: E402

B = 1999


def wild_cluster_p(d: pd.DataFrame, y: str, fe: list[str], cluster: str = "year", seed: int = 0) -> dict:
    """Restricted wild-cluster bootstrap p-value for H0: slope on decade = 0 (Cameron, Gelbach & Miller 2008)."""
    d = d.dropna(subset=[y])
    w = np.ones(len(d))
    yt, xt = _demean([d[y].to_numpy(float), dec(d["year"])], [d[f].to_numpy() for f in fe], w)
    sxx = float((xt * xt).sum())
    codes = pd.factorize(d[cluster])[0]
    G = codes.max() + 1

    def t_stat(yv):
        b = float((xt * yv).sum()) / sxx
        e = yv - b * xt
        score = np.bincount(codes, weights=xt * e, minlength=G)
        se = np.sqrt((score ** 2).sum() * G / (G - 1)) / sxx
        return b, b / se

    b_obs, t_obs = t_stat(yt)
    e0 = yt                       # residuals under H0 (slope = 0): FE-only model
    rng = np.random.default_rng(seed)
    t_star = np.empty(B)
    for i in range(B):
        v = rng.choice((-1.0, 1.0), size=G)
        t_star[i] = t_stat(e0 * v[codes])[1]
    return {"slope_pp_dec": 100 * b_obs, "t": t_obs, "p_wild_cluster": float((np.abs(t_star) >= abs(t_obs)).mean()),
            "n_clusters": int(G), "B": B}


def logit_ame(d: pd.DataFrame, y: str, topic_fe: bool) -> dict:
    d = d.dropna(subset=[y])
    X = pd.DataFrame({"dec": dec(d["year"])}, index=d.index)
    if topic_fe:
        X = X.join(pd.get_dummies(d["topic_label"], prefix="t", drop_first=True, dtype=float))
    X = sm.add_constant(X)
    fit = sm.Logit(d[y].astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d["publication"])[0]})
    p = fit.predict(X)
    scale = float((p * (1 - p)).mean())
    b, se = float(fit.params["dec"]), float(fit.bse["dec"])
    return {"slope_pp_dec": 100 * scale * b, "ci_lo": 100 * scale * (b - 1.96 * se), "ci_hi": 100 * scale * (b + 1.96 * se),
            "note": "logit AME of one decade, SE clustered by newspaper (delta-scaled)"}


def main() -> int:
    df = load()
    base = df[df["ner_ok"] & df["f_hp"].notna()]
    rows = []
    for est, d in (("H1_female_share", base), ("H2_female_share_in_AUTHORITY_B", base[base["B_AUTHORITY"]])):
        for fe_name, fe in (("none", []), ("topic+paper", ["topic_label", "publication"])):
            r = wild_cluster_p(d, "f_hp", fe)
            lp = lpm_trend(d, "f_hp", fe=fe)
            rows.append({"check": "wild_cluster_bootstrap_year", "estimand": est, "spec": fe_name, **r,
                         "ci_lo": lp["ci_lo"], "ci_hi": lp["ci_hi"], "p_analytic_twoway": lp["p"]})
            print(f"WCR {est} {fe_name}: slope={r['slope_pp_dec']:+.2f} p_wild={r['p_wild_cluster']:.3f} "
                  f"(analytic two-way p={lp['p']:.3f})", flush=True)
        for topic_fe in (False, True):
            a = logit_ame(d, "f_hp", topic_fe)
            lp = lpm_trend(d, "f_hp", fe=["topic_label"] if topic_fe else [], cluster="publication", cluster2=None)
            rows.append({"check": "logit_vs_lpm", "estimand": est, "spec": "topic" if topic_fe else "none", **a,
                         "lpm_slope_pp_dec": lp["slope_pp_dec"]})
            print(f"LOGIT {est} topic_fe={topic_fe}: AME={a['slope_pp_dec']:+.2f} vs LPM={lp['slope_pp_dec']:+.2f}", flush=True)

    ts = pd.read_parquet(RESULTS / "trend_summary.parquet")
    fam = ts[ts["estimand"] == "female_share_in_role"].copy()
    fam["q_bh"] = bh_fdr(fam["p"].to_numpy())
    for _, r in fam.iterrows():
        rows.append({"check": "bh_fdr_role_trends", "estimand": "female_share_in_role", "spec": f"{r['role']}|{r['method']}",
                     "slope_pp_dec": r["slope_pp_dec"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_hac": r["p"],
                     "q_bh": r["q_bh"]})
    print(f"BH family: {len(fam)} role-trend tests; q<.05: {int((fam['q_bh'] < 0.05).sum())}; p<.05: {int((fam['p'] < 0.05).sum())}")
    out = pd.DataFrame(rows)
    safe_write(lambda t: out.to_parquet(t, index=False), RESULTS / "reviewer_checks.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
