#!/usr/bin/env python
"""make figures -- publication figures, generated only from results/*.

Mark specs follow the dataviz method (2px round lines, >=8px end dots with a 2px
surface ring, CI as a ~12% wash, solid hairline grid, text in ink never series colour,
a legend whenever >=2 series plus selective direct labels) and the portfolio rule that
every series also carries a dash pattern.  No dual axes: different measures get
separate panels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from gna.lexicons import ROLES  # noqa: E402
from gna.paths import CONFIG, FIGURES, RESULTS  # noqa: E402

P = json.load(open(CONFIG / "palette.json"))["light"]
W, M, N = P["women"], P["men"], P["neutral"]
DASH = {1: (0, ()), 2: (0, (6, 4)), 3: (0, (1, 4)), 4: (0, (10, 4, 2, 4)), 5: (0, (2, 2))}
SEQ = LinearSegmentedColormap.from_list("seq", P["seq"])
ROLE_LABEL = {"PUBLIC_OFFICE": "Public office", "MILITARY": "Military", "BUSINESS": "Business", "LABOR": "Labor",
              "PROFESSIONAL": "Professional", "ARTS_SPORTS": "Arts & sports", "CIVIC": "Civic", "SOCIAL": "Social",
              "FAMILY": "Family", "CRIME_ACCIDENT": "Crime & accident", "AUTHORITY": "Authority (composite)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": P["primary"], "axes.labelcolor": P["secondary"], "axes.labelsize": 9,
    "axes.facecolor": P["surface"], "figure.facecolor": P["surface"], "savefig.facecolor": P["surface"],
    "axes.edgecolor": P["baseline"], "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": P["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": P["muted"], "ytick.color": P["muted"], "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "ytick.left": False, "text.color": P["primary"], "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "lines.dash_capstyle": "round", "lines.scale_dashes": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})
SOURCE = "Source: AmericanStories (Dell et al. 2023, CC-BY-4.0), Chronicling America. Sample of 10,000 articles per year."


def rd(name: str) -> pd.DataFrame | None:
    p = RESULTS / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def save(fig, name: str, note: str = SOURCE) -> None:
    if note:
        # place the source line under everything already drawn (tick labels, x-labels), never on top of it
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.text(bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight() - 0.015, note, fontsize=7.5,
                 color=P["muted"], ha="left", va="top", transform=fig.transFigure)
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.svg")
    plt.close(fig)
    print("wrote", name)


def pct(ax, lo=None, hi=None):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    if lo is not None:
        ax.set_ylim(lo, hi)


def series(ax, x, y, color, dash=1, lo=None, hi=None, label=None, end_label=True, lw=2.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
    ax.plot(x, y, color=color, linestyle=DASH[dash], lw=lw, label=label)
    ok = ~np.isnan(y)
    if ok.any():
        ax.plot(x[ok][-1], y[ok][-1], "o", ms=8, color=color, mec=P["surface"], mew=2, zorder=5)
        if end_label and label:
            ax.annotate(label, (x[ok][-1], y[ok][-1]), xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=P["secondary"], annotation_clip=False)


def years_axis(ax):
    ax.set_xlim(1897, 1966)
    ax.set_xticks([1900, 1915, 1930, 1945, 1960])


# ----------------------------------------------------------------------------- figures
def fig_corpus():
    cs = rd("corpus_summary")
    if cs is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 6.2), sharex=True, gridspec_kw={"hspace": 0.45})
    a = axes[0]
    a.bar(cs["year"], cs["n_publications"], width=1.6, color=N)
    a.set_title("Newspapers contributing to the yearly sample")
    for y, v in zip(cs["year"], cs["n_publications"]):
        if y in (1900, 1921, 1924, 1963):
            a.annotate(f"{v}", (y, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
                       color=P["secondary"])
    series(axes[1], cs["year"], cs["evening_star_share"], N, label=None)
    pct(axes[1], 0, 0.7)
    axes[1].set_title("Share of sampled articles from the Washington Evening Star")
    series(axes[2], cs["year"], cs["share_legible"], N, label="legible", end_label=False)
    series(axes[2], cs["year"], cs["dict_rate_median"], N, dash=2, label="dictionary rate", end_label=False)
    pct(axes[2], 0.6, 1.0)
    axes[2].set_title("OCR quality proxies (article share labelled legible; median dictionary-word rate)")
    axes[2].legend(loc="lower left", ncols=2)
    for ax in axes:
        ax.axvline(1922.5, color=P["baseline"], lw=1)
        years_axis(ax)
    axes[0].annotate("1923: copyright cliff in digitized coverage", (1923.5, axes[0].get_ylim()[1] * 0.9),
                     fontsize=8, color=P["secondary"])
    save(fig, "fig01_corpus")


def fig_coverage():
    ex = rd("extraction_summary")
    if ex is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for dash, (col, lab) in enumerate([("unknown_share_h", "honorific only (h)"),
                                       ("unknown_share_hp", "+ pronoun rule (hp, primary)"),
                                       ("unknown_share_hpn", "+ gendered nouns (hpn)")], start=1):
        series(ax, ex["year"], ex[col], N, dash=dash, label=lab)
    pct(ax, 0, 1)
    years_axis(ax)
    ax.set_title("Person entities with no textual gender signal (UNKNOWN)")
    ax.legend(loc="lower left", ncols=3)
    save(fig, "fig02_unknown_share")


def fig_h1():
    rt, ex = rd("role_trends"), rd("extraction_summary")
    if rt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), gridspec_kw={"hspace": 0.5, "height_ratios": [3, 2]})
    ax = axes[0]
    base = rt[(rt["estimand"] == "female_share") & (rt["population"] == "ner")]
    for dash, tier, lab in ((2, "h", "honorific only"), (3, "hpn", "+ gendered nouns"), (1, "hp", "primary (hp)")):
        d = base[base["tier"] == tier].sort_values("year")
        kw = dict(lo=d["ci_lo"], hi=d["ci_hi"]) if tier == "hp" else {}
        series(ax, d["year"], d["share"], W, dash=dash, label=lab, lw=2 if tier == "hp" else 1.5, **kw)
    pct(ax)
    years_axis(ax)
    ax.set_title("Women's share of gender-signalled person entities")
    ax.legend(loc="lower right", ncols=1)
    if ex is not None:
        a2 = axes[1]
        a2.fill_between(ex["year"], ex["female_share_lower_bound"], ex["female_share_upper_bound"], color=W, alpha=0.12,
                        lw=0)
        series(a2, ex["year"], ex["female_share_upper_bound"], W, dash=2, label="if every UNKNOWN were a woman", lw=1.5)
        series(a2, ex["year"], ex["female_share_lower_bound"], W, dash=3, label="if every UNKNOWN were a man", lw=1.5)
        pct(a2, 0, 1)
        years_axis(a2)
        a2.set_title("Women's share of ALL person entities, under the two extreme treatments of UNKNOWN")
    save(fig, "fig03_female_share")


def _grid(n, ncols=4, h=1.9):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, h * nrows + 0.6), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.12})
    return fig, np.atleast_1d(axes).ravel()


def fig_roles():
    rt = rd("role_trends")
    if rt is None:
        return
    roles = ["AUTHORITY"] + ROLES
    fig, axes = _grid(len(roles))
    for ax, r in zip(axes, roles):
        for m, dash, lab in (("A", 2, "A lexical window"), ("B", 1, "B dependency rules")):
            d = rt[(rt["estimand"] == "female_share_in_role") & (rt["role"] == r) & (rt["method"] == m)].sort_values("year")
            d = d[d["n"] >= 30]
            series(ax, d["year"], d["share"], W if m == "B" else N, dash=dash, label=lab, end_label=False,
                   lo=d["ci_lo"] if m == "B" else None, hi=d["ci_hi"] if m == "B" else None, lw=1.8)
        ax.set_title(ROLE_LABEL[r], fontsize=9)
        pct(ax, 0, 1)
        ax.set_xticks([1900, 1930, 1960])
    for ax in axes[len(roles):]:
        ax.set_visible(False)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Women's share of gender-signalled people in each role, by role method", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    save(fig, "fig04_roles_female_share")


def fig_role_rates():
    rt = rd("role_trends")
    if rt is None:
        return
    fig, axes = _grid(len(ROLES) + 1)
    for ax, r in zip(axes, ["AUTHORITY"] + ROLES):
        for g, col, dash, lab in (("F", W, 1, "women"), ("M", M, 2, "men")):
            d = rt[(rt["estimand"] == f"role_rate_{g}") & (rt["role"] == r) & (rt["method"] == "B")].sort_values("year")
            series(ax, d["year"], d["share"], col, dash=dash, label=lab, end_label=False, lw=1.8)
        ax.set_title(ROLE_LABEL[r], fontsize=9)
        pct(ax)
        ax.set_xticks([1900, 1930, 1960])
    for ax in axes[len(ROLES) + 1:]:
        ax.set_visible(False)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Share of women and of men placed in each role (method B)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold")
    save(fig, "fig05_role_rates_by_gender")


def fig_adjusted():
    ta = rd("topic_adjusted")
    if ta is None:
        return
    lp = ta[ta["kind"] == "lpm"]
    outs = [("female_share", "All gender-signalled people"), ("female_share_in_AUTHORITY_B", "Authority roles (B)")]
    specs = ["none", "topic", "paper", "topic+paper", "topic+paper+page"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (o, t) in zip(axes, outs):
        for off, samp, filled, lab in ((-0.12, "all", True, "all newspapers"), (0.12, "excl_evening_star", False,
                                                                                  "excluding Evening Star")):
            d = lp[(lp["outcome"] == o) & (lp["sample"] == samp)].set_index("fe").reindex(specs)
            y = np.arange(len(specs)) + off
            ax.hlines(y, d["ci_lo"], d["ci_hi"], color=W, lw=2)
            ax.plot(d["slope_pp_dec"], y, "o", ms=8, mfc=W if filled else P["surface"], mec=W, mew=2, label=lab)
        ax.axvline(0, color=P["baseline"], lw=1)
        ax.set_yticks(range(len(specs)), ["raw (no FE)", "topic FE", "newspaper FE", "topic + newspaper",
                                          "+ page position"])
        ax.invert_yaxis()
        ax.set_title(t)
        ax.set_xlabel("trend, percentage points per decade (95% CI)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=8)
    save(fig, "fig06_raw_vs_adjusted")


def fig_decomp():
    ta = rd("topic_adjusted")
    if ta is None:
        return
    dc = ta[(ta["kind"] == "decomposition")]
    labels = {"topic_label": "topic", "section": "section", "paper_group": "Evening Star vs other"}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (o, t) in zip(axes, [("female_share", "All gender-signalled people"),
                                 ("female_share_in_AUTHORITY_B", "Authority roles (B)")]):
        d = dc[dc["outcome"] == o]
        rows = []
        for _, r in d.iterrows():
            for comp in ("total", "composition", "within"):
                rows.append((f"{labels.get(r['group'], r['group'])}: {comp}", r[f"{comp}_pp"], r[f"{comp}_lo"],
                             r[f"{comp}_hi"], comp))
        y = np.arange(len(rows))
        for yi, (lab, v, lo, hi, comp) in zip(y, rows):
            ax.barh(yi, v, height=0.55, color=W if comp == "within" else (N if comp == "total" else P["baseline"]))
            ax.hlines(yi, lo, hi, color=P["primary"], lw=1)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=7.8)
        ax.invert_yaxis()
        ax.axvline(0, color=P["baseline"], lw=1)
        ax.set_title(t)
        ax.set_xlabel("change from 1900–21 to 1948–63, percentage points")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    save(fig, "fig07_decomposition")


def fig_methods():
    ma = rd("method_agreement")
    if ma is None:
        return
    yr = ma[(ma["kind"] == "yearly") & (ma["role"] == "AUTHORITY")]
    methods = [m for m in ["A", "B", "C", "D", "E"] if m in set(yr["method"])]
    names = {"A": "A · lexical", "B": "B · dependency", "C": "C · bag-of-words", "D": "D · embeddings", "E": "E · LLM"}
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(11, 2.8), gridspec_kw={"wspace": 0.22,
                                                                                    "width_ratios": [1] * len(methods) + [1.3]})
    for ax, m in zip(axes, methods):
        d = yr[yr["method"] == m].sort_values("year")
        if m == "E":                              # thin LLM subsample: pool the four periods
            per = pd.cut(d["year"], [1899, 1915, 1930, 1945, 1963])
            d = d.groupby(per, observed=True).agg(year=("year", "mean"), n=("n", "sum"), k=("k", "sum"))
            d = d.assign(share=d["k"] / d["n"])
            ax.plot(d["year"], d["share"], "o", ms=5, color=W)
        else:
            d = d[d["n"] >= 20]
        series(ax, d["year"], d["share"], W, label=None)
        ax.set_title(names[m] + (" (4 periods)" if m == "E" else ""), fontsize=8.5, loc="left")
        pct(ax, 0, 0.6)
        ax.set_xticks([1900, 1930, 1960])
        if ax is not axes[0]:
            ax.set_yticklabels([])
    ax = axes[-1]
    tr = ma[(ma["kind"] == "trend") & (ma["role"] == "AUTHORITY") & ma["method"].isin(methods)].set_index("method").reindex(methods)
    yy = np.arange(len(methods))
    ax.hlines(yy, tr["ci_lo"], tr["ci_hi"], color=W, lw=2)
    ax.plot(tr["slope_pp_dec"], yy, "o", ms=8, color=W, mec=P["surface"], mew=2)
    ax.axvline(0, color=P["baseline"], lw=1)
    ax.set_yticks(yy, methods)
    ax.invert_yaxis()
    ax.set_title("Trend, pp/decade", fontsize=8.5)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Women's share of authority roles on the same entities, measured five ways", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold", y=1.06)
    save(fig, "fig08_method_comparison")


def heat(ax, mat: pd.DataFrame, fmt, vmin=0, vmax=1):
    im = ax.imshow(mat.to_numpy(float), cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.notna(v):
                frac = (v - vmin) / (vmax - vmin)
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if frac > 0.55 else P["primary"])
    ax.set_xticks(range(mat.shape[1]), mat.columns, fontsize=7.8)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7.8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_validation():
    v = rd("validation")
    if v is None:
        return
    methods = [m for m in ["A_lexical", "B_dependency", "C_bow@0.5", "D_embed@0.5", "E_llm"] if m in set(v["method"])]
    roles = ["AUTHORITY"] + ROLES
    mat = v[v["method"].isin(methods) & v["target"].isin(roles)].pivot_table(index="method", columns="target", values="f1")
    mat = mat.reindex(index=methods, columns=roles)
    mat.columns = [ROLE_LABEL[c].replace(" (composite)", "") for c in mat.columns]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    heat(ax, mat, lambda x: f"{x:.2f}")
    ax.set_title("Role F1 against v1 reference labels (sampling-weighted; reference labels are AI-produced, not human)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "fig09_validation_f1")


def fig_robust():
    rb = rd("robustness")
    if rb is None:
        return
    c = rb[rb["kind"] == "cell"].dropna(subset=["slope_pp_dec"])
    groups = [("H1_female_share", "-", "H1 all people"), ("H2_female_share_in_AUTHORITY", "A", "H2 authority · A"),
              ("H2_female_share_in_AUTHORITY", "B", "H2 authority · B")]
    for pre in ("C", "D"):
        for th in (0.3, 0.5, 0.7):
            groups.append(("H2_female_share_in_AUTHORITY", f"{pre}@{th}", f"H2 authority · {pre} @{th}"))
    groups.append(("female_share_in_CIVIC", "B", "Civic roles · B"))
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(groups) + 1.2))
    rng = np.random.default_rng(0)
    for i, (e, m, lab) in enumerate(groups):
        d = c[(c["estimand"] == e) & (c["method"] == m)]
        if d.empty:
            continue
        ax.plot(d["slope_pp_dec"], i + rng.uniform(-0.18, 0.18, len(d)), "o", ms=3.2, color=W, alpha=0.45, mew=0)
        ax.annotate(f"{int((d['slope_pp_dec'] > 0).sum()):,} of {len(d):,} > 0", (1.0, i), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.8, color=P["secondary"])
    ax.axvline(0, color=P["baseline"], lw=1)
    ax.set_yticks(range(len(groups)), [g[2] for g in groups])
    ax.invert_yaxis()
    ax.set_xlabel("trend estimate across specifications, percentage points per decade")
    ax.set_title("Multiverse: every specification's trend estimate")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, "fig10_robustness")


def fig_topics():
    ta = rd("topic_adjusted")
    if ta is None:
        return
    gp = ta[(ta["kind"] == "group_period") & (ta["group"] == "topic_label")]
    mat = gp.pivot_table(index="group_value", columns="period", values="female_share")
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(5.8, 0.28 * len(mat) + 1.2))
    heat(ax, mat, lambda x: f"{100 * x:.0f}", vmin=0, vmax=max(0.6, float(np.nanmax(mat.to_numpy()))))
    ax.set_title("Women's share (%) of gender-signalled people, by topic and period")
    save(fig, "fig11_topics")


def fig_quotes():
    qt = rd("quote_trends")
    if qt is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    d = qt[qt["estimand"] == "female_share_mentions"].sort_values("year")
    series(ax, d["year"], d["share"], N, dash=2, lo=d["ci_lo"], hi=d["ci_hi"], label="all gender-signalled people")
    d = qt[(qt["estimand"] == "female_share_speakers") & (qt["attribution"] == "name")].sort_values("year")
    series(ax, d["year"], d["share"], W, dash=1, lo=d["ci_lo"], hi=d["ci_hi"], label="people quoted (by name)")
    pct(ax)
    years_axis(ax)
    ax.set_title("Women's share among people mentioned vs people quoted")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), ncols=1)
    save(fig, "fig12_quotes")


def fig_agency():
    lg = rd("language")
    if lg is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for g, col, dash, lab in (("F", W, 1, "women"), ("M", M, 2, "men")):
        d = lg[(lg["kind"] == "agency_yearly") & (lg["gender"] == g)].sort_values("year")
        series(ax, d["year"], d["share"], col, dash=dash, lo=d["ci_lo"], hi=d["ci_hi"], label=lab)
    pct(ax)
    years_axis(ax)
    ax.set_title("Share of syntactic positions that are agent (subject of an action) rather than patient")
    ax.legend(loc="lower left", ncols=2)
    save(fig, "fig13_agency")


def fig_kappa():
    ma = rd("method_agreement")
    if ma is None:
        return
    k = ma[(ma["kind"] == "kappa") & (ma["role"] == "AUTHORITY") & (ma["period"] != "all")]
    k = k.assign(pair=k["method_a"] + "–" + k["method_b"])
    mat = k.pivot_table(index="pair", columns="period", values="kappa")
    fig, ax = plt.subplots(figsize=(5.2, 0.34 * len(mat) + 1.2))
    heat(ax, mat, lambda x: f"{x:.2f}", vmin=0, vmax=1)
    ax.set_title("Agreement between role methods on 'authority' (Cohen's κ) by period")
    save(fig, "fig14_kappa")


def main() -> int:
    for f in (fig_corpus, fig_coverage, fig_h1, fig_roles, fig_role_rates, fig_adjusted, fig_decomp, fig_methods,
              fig_validation, fig_robust, fig_topics, fig_quotes, fig_agency, fig_kappa):
        try:
            f()
        except Exception as e:  # a missing/partial result must not block the other figures
            print(f"[skip] {f.__name__}: {e!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
