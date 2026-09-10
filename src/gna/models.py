"""Temporal estimation: bootstrap shares, HAC trends, fixed-effects LPM with robust SEs.

Conventions
-----------
* Time enters as ``dec = (year - 1930) / 10`` so slopes are **per decade**; shares are
  proportions and slopes are reported x100 as **percentage points per decade**.
* Annual points are never treated as i.i.d.: the aggregate trend uses Newey-West HAC
  standard errors, and the micro model clusters by newspaper and, two-way, by year.
* Raw-vs-adjusted comparisons use the same estimator (linear probability model),
  with and without absorbed fixed effects.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

CENTER_YEAR = 1930


def dec(year) -> np.ndarray:
    return (np.asarray(year, dtype=float) - CENTER_YEAR) / 10.0


# --------------------------------------------------------------------------- shares
def yearly_share(df: pd.DataFrame, success: str, cluster: str = "article_id", by: str = "year",
                 B: int = 400, seed: int = 0, weight: str | None = None) -> pd.DataFrame:
    """Share of rows with `success` per `by`, with a cluster (Poisson) bootstrap 95% CI.

    Clusters (articles) are resampled with Poisson(1) weights; entities inside an
    article move together, which respects within-article dependence.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(by, sort=True):
        w = g[weight].to_numpy(float) if weight else np.ones(len(g))
        agg = pd.DataFrame({"c": g[cluster].to_numpy(), "s": g[success].to_numpy(float) * w, "n": w}) \
            .groupby("c", sort=False).sum()
        s, n = agg["s"].to_numpy(), agg["n"].to_numpy()
        share = s.sum() / n.sum() if n.sum() else np.nan
        if len(agg) > 1 and n.sum() > 0:
            pw = rng.poisson(1.0, size=(B, len(agg)))
            den = pw @ n
            bs = np.divide(pw @ s, den, out=np.full(B, np.nan), where=den > 0)
            lo, hi = np.nanquantile(bs, [0.025, 0.975])
        else:
            lo = hi = np.nan
        rows.append({by: key, "n": float(n.sum()), "n_clusters": len(agg), "k": float(s.sum()),
                     "share": share, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def hac_trend(yearly: pd.DataFrame, value: str = "share", weight: str = "n", maxlags: int = 2,
              year_col: str = "year") -> dict:
    """WLS of the yearly series on decade with Newey-West HAC SEs.  Returns pp/decade."""
    d = yearly.dropna(subset=[value])
    d = d[d[weight] > 0]
    if len(d) < 4:
        return {"slope_pp_dec": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n_years": len(d)}
    X = sm.add_constant(dec(d[year_col]))
    fit = sm.WLS(d[value].to_numpy(float), X, weights=d[weight].to_numpy(float)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True})
    b, se = fit.params[1], fit.bse[1]
    tcrit = stats.t.ppf(0.975, df=max(len(d) - 2, 1))
    return {"slope_pp_dec": 100 * b, "se": 100 * se, "ci_lo": 100 * (b - tcrit * se),
            "ci_hi": 100 * (b + tcrit * se), "p": float(2 * stats.t.sf(abs(b / se), df=max(len(d) - 2, 1))),
            "level_1930": 100 * fit.params[0], "n_years": len(d)}


# --------------------------------------------------------------------------- FE LPM
def _demean(arrs: list[np.ndarray], groups: list[np.ndarray], w: np.ndarray, tol: float = 1e-9,
            max_iter: int = 200) -> list[np.ndarray]:
    """Absorb any number of fixed effects by weighted alternating projections."""
    out = [a.astype(float).copy() for a in arrs]
    if not groups:
        return [a - np.average(a, weights=w) for a in out]
    codes = [pd.factorize(g)[0] for g in groups]
    sizes = [np.bincount(c, weights=w) for c in codes]
    for _ in range(max_iter):
        delta = 0.0
        for c, sz in zip(codes, sizes):
            for i, a in enumerate(out):
                m = np.bincount(c, weights=a * w) / np.where(sz > 0, sz, 1)
                adj = m[c]
                delta = max(delta, float(np.abs(adj).max()))
                out[i] = a - adj
        if delta < tol:
            break
    return out


def _cluster_var(xt: np.ndarray, e: np.ndarray, w: np.ndarray, cl: np.ndarray) -> tuple[float, int]:
    score = pd.Series(xt * e * w).groupby(pd.factorize(cl)[0]).sum().to_numpy()
    G = len(score)
    return float((score ** 2).sum()) * (G / max(G - 1, 1)), G


def lpm_trend(df: pd.DataFrame, y: str, fe: list[str] | None = None, cluster: str = "publication",
              cluster2: str | None = "year", weight: str | None = None) -> dict:
    """Linear probability model y ~ dec + FE; slope in pp/decade with (two-way) cluster SEs."""
    d = df.dropna(subset=[y]).copy()
    w = d[weight].to_numpy(float) if weight else np.ones(len(d))
    x = dec(d["year"])
    yy = d[y].to_numpy(float)
    groups = [d[f].to_numpy() for f in (fe or [])]
    yt, xt = _demean([yy, x], groups, w)
    sxx = float((w * xt * xt).sum())
    if sxx <= 0:
        return {"slope_pp_dec": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": len(d)}
    b = float((w * xt * yt).sum()) / sxx
    e = yt - b * xt
    v1, g1 = _cluster_var(xt, e, w, d[cluster].to_numpy())
    var, G = v1, g1
    if cluster2:
        v2, g2 = _cluster_var(xt, e, w, d[cluster2].to_numpy())
        v12, _ = _cluster_var(xt, e, w, (d[cluster].astype(str) + "|" + d[cluster2].astype(str)).to_numpy())
        var, G = max(v1 + v2 - v12, v1, v2), min(g1, g2)
    se = math.sqrt(var) / sxx
    tcrit = stats.t.ppf(0.975, df=max(G - 1, 1))
    return {"slope_pp_dec": 100 * b, "se": 100 * se, "ci_lo": 100 * (b - tcrit * se), "ci_hi": 100 * (b + tcrit * se),
            "p": float(2 * stats.t.sf(abs(b / se), df=max(G - 1, 1))) if se > 0 else np.nan,
            "n": int(len(d)), "n_clusters": int(G), "mean_y": float(np.average(yy, weights=w)),
            "fe": "+".join(fe or []) or "none"}


# --------------------------------------------------------------------------- misc
def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaNs preserved)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def unknown_bounds(n_f: float, n_m: float, n_u: float) -> dict:
    """Share F among all entities under the extreme allocations of UNKNOWN."""
    tot = n_f + n_m + n_u
    if tot == 0:
        return {"share_f_classified": np.nan, "lower": np.nan, "upper": np.nan}
    return {"share_f_classified": n_f / (n_f + n_m) if (n_f + n_m) else np.nan,
            "lower": n_f / tot, "upper": (n_f + n_u) / tot}


def log_odds_dirichlet(counts_a: pd.Series, counts_b: pd.Series, prior: pd.Series | None = None,
                       alpha0: float = 1000.0) -> pd.DataFrame:
    """Monroe, Colaresi & Quinn (2008) log-odds ratio with informative Dirichlet prior.

    Returns delta (a vs b), its variance, z-score and two-sided p per word.
    """
    vocab = counts_a.index.union(counts_b.index)
    ya = counts_a.reindex(vocab, fill_value=0).astype(float)
    yb = counts_b.reindex(vocab, fill_value=0).astype(float)
    pr = (ya + yb) if prior is None else prior.reindex(vocab, fill_value=0).astype(float)
    a = alpha0 * (pr + 0.01) / (pr + 0.01).sum()
    na, nb, a0 = ya.sum(), yb.sum(), a.sum()
    la = np.log((ya + a) / (na + a0 - ya - a))
    lb = np.log((yb + a) / (nb + a0 - yb - a))
    delta = la - lb
    var = 1.0 / (ya + a) + 1.0 / (yb + a)
    z = delta / np.sqrt(var)
    return pd.DataFrame({"n_a": ya, "n_b": yb, "delta": delta, "var": var, "z": z,
                         "p": 2 * stats.norm.sf(np.abs(z))}).sort_values("z")
