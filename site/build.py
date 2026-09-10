#!/usr/bin/env python
"""make site (step 2) -- render every route to static HTML from site/public/data/*.json.

Zero dependencies (stdlib only).  Every number on every page is read from the exported
results; prose that states a direction ("rose", "fell") is chosen by the sign of an
estimate AND whether its 95% CI excludes zero, never typed.  A missing value renders as
an em dash, never as a guess.
"""
from __future__ import annotations

import html
import json
import math
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "public" / "data"
ASSETS = ROOT / "site" / "assets"
DIST = ROOT / "site" / "dist"

E = html.escape
MINUS = "−"
NAV = [("/", "Overview"), ("/timeline/", "Timeline"), ("/roles/", "Roles"), ("/quotes/", "Quotes"),
       ("/language/", "Language"), ("/topics/", "Topics"), ("/methods/", "Methods"),
       ("/disagreement/", "Disagreement"), ("/validation/", "Validation"), ("/robustness/", "Robustness"),
       ("/failures/", "Failures"), ("/data/", "Data"), ("/paper/", "Paper")]
ROLES = ["PUBLIC_OFFICE", "MILITARY", "BUSINESS", "LABOR", "PROFESSIONAL", "ARTS_SPORTS", "CIVIC", "SOCIAL",
         "FAMILY", "CRIME_ACCIDENT"]
ROLE_LABEL = {"ALL": "All gender-signalled people", "AUTHORITY": "Authority (public office, business, professional)",
              "PUBLIC_OFFICE": "Public office", "MILITARY": "Military", "BUSINESS": "Business", "LABOR": "Labor",
              "PROFESSIONAL": "Professional & expert", "ARTS_SPORTS": "Arts & sports", "CIVIC": "Civic organisations",
              "SOCIAL": "Society pages", "FAMILY": "Family (kin-identified)", "CRIME_ACCIDENT": "Crime & accident"}
ROLE_DEF = {
    "PUBLIC_OFFICE": "Elected or appointed officials, diplomats, judges, police and law enforcement, candidates.",
    "MILITARY": "Officers and enlisted personnel. Kept apart so wartime years do not swamp public office.",
    "BUSINESS": "Owners, executives, managers, merchants, bankers, brokers.",
    "LABOR": "Employees and wage workers, domestic service, farm and industrial work.",
    "PROFESSIONAL": "Medicine, science, education, law practice, clergy, journalism and authorship, named experts.",
    "ARTS_SPORTS": "Performers, artists, athletes, coaches.",
    "CIVIC": "Officers and members of clubs, societies, churches, unions, parties and charities.",
    "SOCIAL": "Society-page roles: hostess, host, guest, debutante, bride, patroness.",
    "FAMILY": "Identified through a kin relation (“wife of”, “widow of”, “son of”).",
    "CRIME_ACCIDENT": "Victims, suspects, defendants; subjects of passive harm or arrest predicates.",
}
METHOD_NAME = {"A": "A · lexical window", "B": "B · dependency rules", "C": "C · bag-of-words classifier",
               "D": "D · contextual embeddings", "E": "E · LLM (Qwen3-8B)"}
METHOD_DEF = {
    "A": "Any role noun within ten tokens of the name, in the same sentence. One sense per word.",
    "B": "Titles on the name, appositives, “was elected president of…”, organisational heads typed by the organisation they head, passive harm predicates. Never inherits a relative’s role.",
    "C": "Logistic regression on the words around the person (name masked), trained on reference labels and cross-fitted.",
    "D": "MiniLM token states pooled over the name and its context, with a logistic head; cross-fitted on reference labels.",
    "E": "An open-weights language model reading the same context under a fixed JSON prompt at temperature 0. One measurement among five, never ground truth.",
}


# ----------------------------------------------------------------------------- data + formatting
def load(name, default=None):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else default


def ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def pct(v, d=1):
    return f"{100 * v:.{d}f}%" if ok(v) else "—"


def signed(v, d=2, unit=""):
    if not ok(v):
        return "—"
    s = "+" if v > 0 else (MINUS if v < 0 else "")
    return f"{s}{abs(v):.{d}f}{unit}"


def num(v):
    return f"{int(round(v)):,}" if ok(v) else "—"


def dec(v, d=2):
    return f"{v:.{d}f}".replace("-", MINUS) if ok(v) else "—"


def ci(lo, hi, f=lambda x: signed(x)):
    return f"95% CI {f(lo)} to {f(hi)}" if ok(lo) and ok(hi) else "CI not available"


def trend_words(slope, lo, hi, up="rose", down="fell"):
    if not (ok(slope) and ok(lo) and ok(hi)):
        return "cannot be assessed"
    if lo > 0:
        return up
    if hi < 0:
        return down
    return "shows no trend distinguishable from zero"


def where(rows, **kw):
    return [r for r in (rows or []) if all(r.get(k) == v for k, v in kw.items())]


def first(rows, **kw):
    r = where(rows, **kw)
    return r[0] if r else {}


# ----------------------------------------------------------------------------- html helpers
def table(headers, rows, numeric=(), caption=None):
    th = "".join(f'<th scope="col"{" class=n" if i in numeric else ""}>{E(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td{" class=n" if i in numeric else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    cap = f"<caption>{E(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def figure(kind, spec, title, desc="", table_html="", source="", variant=None):
    js = json.dumps(spec, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    var = f' data-variant="{E(json.dumps(variant))}"' if variant else ""
    return (f'<figure class="viz" data-viz="{kind}"{var}>'
            f'<figcaption><span class="t">{E(title)}</span>{f"<span class=d>{desc}</span>" if desc else ""}</figcaption>'
            f'<div class="viz-plot"></div><script type="application/json">{js}</script>'
            f'<noscript><p class="viz-empty">This chart draws with JavaScript. Every value is in the table view below.</p></noscript>'
            f'<details class="tableview"><summary>Table view</summary><div class="table-scroll">{table_html}</div></details>'
            f'{f"<p class=viz-source>{source}</p>" if source else ""}</figure>')


def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_json(v) for v in o]
    if isinstance(o, float) and not ok(o):
        return None
    return o


def series(rows, label, color, dash=1, min_n=0, ci_on=True, width=None):
    pts = []
    for r in sorted(rows, key=lambda r: r["year"]):
        if (r.get("n") or 0) < min_n or not ok(r.get("share")):
            continue
        p = {"x": r["year"], "y": r["share"], "n": r.get("n")}
        if ci_on and ok(r.get("ci_lo")) and ok(r.get("ci_hi")):
            p["lo"], p["hi"] = r["ci_lo"], r["ci_hi"]
        pts.append(p)
    s = {"label": label, "color": color, "dash": dash, "points": pts}
    if width:
        s["width"] = width
    return s


def series_table(ss, f=pct):
    years = sorted({p["x"] for s in ss for p in s["points"]})
    head = ["Year"] + [s["label"] for s in ss]
    rows = []
    for y in years:
        row = [str(y)]
        for s in ss:
            p = next((q for q in s["points"] if q["x"] == y), None)
            if not p:
                row.append("—")
                continue
            extra = f" ({f(p['lo'])}–{f(p['hi'])})" if "lo" in p else ""
            n = f" · n={num(p['n'])}" if ok(p.get("n")) else ""
            row.append(f"{f(p['y'])}{extra}{n}")
        rows.append(row)
    return table(head, rows, numeric=set(range(1, len(head))))


def status_mark(s):
    return f'<span class="status" data-s="{E(s)}">{E(s)}</span>'


SOURCE = ("AmericanStories (Dell et al. 2023, CC-BY-4.0), built on Chronicling America (Library of Congress). "
          "10,000 sampled articles per year, 22 years.")


# ----------------------------------------------------------------------------- shell
class Site:
    def __init__(self):
        self.rel = load("release", {}) or {}
        h = self.rel.get("headline", {})
        self.corpus_h = h.get("corpus", {})
        git = self.rel.get("git", {})
        self.sha7 = (git.get("sha") or "unknown")[:7] + ("*" if git.get("dirty") else "")
        self.frozen = (self.rel.get("frozen_utc") or "")[:10]

    def page(self, route, title, desc, body):
        nav = "".join(f'<li><a href="{r}"{" aria-current=page" if r == route else ""}>{E(t)}</a></li>' for r, t in NAV)
        c = self.corpus_h
        issue = (f"<span>Vol. I</span><span>Frozen {E(self.frozen or '—')}</span><span>Git {E(self.sha7)}</span>"
                 f"<span>{num(c.get('n_articles'))} articles · {c.get('first_year', '—')}–{c.get('last_year', '—')}</span>"
                 f'<button class="theme-toggle" type="button" hidden>Theme: system</button>')
        doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)} · GenderNews Atlas</title>
<meta name="description" content="{E(desc)}">
<meta property="og:title" content="{E(title)} · GenderNews Atlas">
<meta property="og:description" content="{E(desc)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Newsreader:opsz,wght@6..72,400;6..72,700&display=swap">
<link rel="stylesheet" href="/assets/design-system/master.css">
<link rel="stylesheet" href="/assets/tokens.css">
<link rel="stylesheet" href="/assets/site.css">
<script>try{{var t=localStorage.getItem("gna-theme");if(t&&t!=="system")document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="mast wrap">
<p class="mast-line">{issue}</p>
<p class="mast-name"><a href="/">GenderNews Atlas</a></p>
<details class="mast-menu" open><summary>Sections</summary>
<nav class="mast-nav" aria-label="Sections"><ul>{nav}</ul></nav></details>
<hr class="mast-rule" aria-hidden="true">
</header>
<main id="main" class="wrap">
{body}
</main>
<footer class="foot wrap">
<p class="wordmark">GenderNews Atlas</p>
<p class="tagline">How robust are NLP measurements of gender in U.S. newspapers, 1900–1963?</p>
<p class="links">Corpus: <a href="https://huggingface.co/datasets/dell-research-harvard/AmericanStories">AmericanStories</a> (Dell et al. 2023, CC-BY-4.0), from <a href="https://chroniclingamerica.loc.gov/">Chronicling America</a>, Library of Congress · An independent project inspired by the Berkeley URAP question on gender roles in news; it does not use that project's Washington Post corpus · <a href="/data/">Data &amp; provenance</a> · <a href="/paper/">Paper</a> · <a href="/annotate/">Annotation tool</a></p>
</footer>
<script type="module" src="/assets/app.js"></script>
</body>
</html>
"""
        out = DIST / route.strip("/") / "index.html" if route != "/" else DIST / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(doc)


def head(title, dek):
    return f'<header class="page-head"><h1>{E(title)}</h1><p class="dek">{dek}</p></header>'


# ----------------------------------------------------------------------------- pages
def home(site: Site):
    ex = load("extraction", []) or []
    ts = load("trend_summary", []) or []
    rt = load("role_trends", []) or []
    ma = load("methods", []) or []
    rs = load("robustness_summary", []) or []
    val = load("validation", []) or []
    claims = (load("claims", {}) or {}).get("claims", [])

    tot = sum(r.get("n_entities_ner") or 0 for r in ex)
    unk = sum((r.get("unknown_share_hp") or 0) * (r.get("n_entities_ner") or 0) for r in ex) / tot if tot else None
    fig_v = round(100 * unk) if ok(unk) else None
    n_art = site.corpus_h.get("n_articles")
    hero = f"""<section class="hero reveal" style="--i:0">
<div><p class="hero-figure tnum" aria-hidden="true"><span data-tick="{fig_v if fig_v is not None else ''}">{fig_v if fig_v is not None else '—'}</span><span class="unit">%</span></p></div>
<div><h1>of the people these newspapers name carry no textual sign of their gender.</h1>
<p class="qualifier">Measured on {num(tot)} people in {num(n_art)} articles sampled from 1900 to 1963. Before anyone asks how the share of women changed, that number decides what the share can mean. This atlas tracks both, and tests whether the answer survives a change of method.</p>
<a class="cta" href="/robustness/">See what survives the robustness grid</a></div>
</section>"""

    h1 = first(ts, estimand="female_share", tier="hp", population="ner", role="ALL")
    mt = [r for r in where(ma, kind="trend", role="AUTHORITY") if r.get("method") in ("A", "B", "C", "D", "E")]
    sl = [r["slope_pp_dec"] for r in mt if ok(r.get("slope_pp_dec"))]
    rh1 = next((r for r in rs if r.get("estimand") == "H1_female_share" and not any(r.get(k) for k in ("method", "adjust", "tier", "papers"))), {})
    vf = [r["f1"] for r in val if r.get("target") == "AUTHORITY" and r.get("method") in ("A_lexical", "B_dependency", "C_bow@0.5", "D_embed@0.5", "E_llm") and ok(r.get("f1"))]
    stats = f"""<ul class="stats reveal" style="--i:1">
<li><span class="v tnum">{pct(h1.get('early_share'))} → {pct(h1.get('late_share'))}</span><span class="l">women's share of gender-signalled people, 1900–21 to 1948–63</span><span class="s">trend {signed(h1.get('slope_pp_dec'))} pp per decade, {ci(h1.get('ci_lo'), h1.get('ci_hi'))}</span></li>
<li><span class="v tnum">{signed(min(sl)) if sl else '—'} to {signed(max(sl)) if sl else '—'}</span><span class="l">pp per decade: the trend in women's share of authority roles, depending on which of {len(sl)} methods measures “authority”</span><span class="s">same people, same years; <a href="/methods/">compare methods</a></span></li>
<li><span class="v tnum">{pct(rh1.get('share_positive'), 0)}</span><span class="l">of {num(rh1.get('n_cells'))} analysis specifications show a rising women's share</span><span class="s">slope range {signed(rh1.get('slope_min'))} to {signed(rh1.get('slope_max'))} pp per decade; <a href="/robustness/">the grid</a></span></li>
<li><span class="v tnum">{dec(min(vf)) if vf else '—'}–{dec(max(vf)) if vf else '—'}</span><span class="l">F1 for “authority” across methods, against reference labels</span><span class="s">reference labels were produced by an AI annotator, not humans; <a href="/validation/">details</a></span></li>
</ul>"""

    cl = "".join(f'<li class="finding"><p class="claim">{E(c["claim"])}</p><p>{status_mark(c["status"])} · {E(c["id"])} · <a href="{E(c["page"])}">evidence</a></p></li>' for c in claims)
    claims_html = (f'<h2>What holds, and what does not</h2><p class="prose">Each hypothesis was written down before any '
                   f'measurement was run. A status of <em>supported</em> means the pre-registered criterion holds in the primary '
                   f'specification and the sign survives every cell of the robustness grid; <em>tentative</em> means only the first; '
                   f'<em>unsupported</em> means neither. Statuses are computed by <code>scripts/claims.py</code>.</p>'
                   f'<ul class="evidence">{cl}</ul>') if claims else ""

    base = where(rt, estimand="female_share", population="ner")
    ss = [series(where(base, tier="hp"), "primary: honorific + pronoun", "women", 1),
          series(where(base, tier="h"), "honorific only", "women", 2, ci_on=False, width=1.5),
          series(where(base, tier="hpn"), "+ gendered nouns", "women", 3, ci_on=False, width=1.5)]
    fig = figure("line", clean_json({"title": "Women's share of gender-signalled people", "series": ss, "y": {"format": "pct0"},
                                     "refLines": [{"x": 1922.5, "label": "1923: digitisation cliff"}]}),
                 "Women's share of gender-signalled people, by gender-evidence rule",
                 "Shaded band: 95% article-cluster bootstrap CI for the primary rule.", series_table(ss), SOURCE)

    index = "".join(f'<li><a href="{r}">{E(t)}</a> — {E(d)}</li>' for r, t, d in [
        ("/timeline/", "Timeline", "the trend, raw and composition-adjusted, with filters"),
        ("/roles/", "Roles", "who is placed in which role, and how that changed"),
        ("/quotes/", "Quotes", "who is quoted, versus who is mentioned"),
        ("/language/", "Language", "agency and description (exploratory)"),
        ("/topics/", "Topics", "topic mix, newspaper mix, and what adjusting for them does"),
        ("/methods/", "Methods", "five ways to measure a role, on the same people"),
        ("/disagreement/", "Disagreement", "the passages where methods disagree"),
        ("/validation/", "Validation", "each method against reference labels"),
        ("/robustness/", "Robustness", "every specification's answer"),
        ("/failures/", "Failures", "where the pipeline breaks"),
        ("/data/", "Data", "corpus, sampling audit, provenance"),
        ("/paper/", "Paper", "the manuscript")])
    body = f"""{hero}{stats}
<section class="two-col"><div>
<h2>The question</h2>
<div class="prose"><p>How did the representation of women and men in American newspapers change between 1900 and 1963, and how much of any answer is an artefact of the NLP method used to measure it?</p>
<p>The second half of the question is the point. A trend line from one pipeline says little on its own. Here every headline estimate is re-measured with different rules for who counts as a person, what counts as evidence of gender, and what counts as a role. Estimates are shown with the corpus's changing composition held fixed and with it left free.</p></div>
{claims_html}
{fig}
</div>
<aside class="aside"><h3>Read the atlas</h3><ul>{index}</ul>
<h3>Scope</h3><p>Gender here is the gender <em>signalled by the text</em> (Mrs., Miss, Mr., pronouns). It is not anyone's identity. First names are never used. People the text does not gender stay UNKNOWN and are reported.</p></aside>
</section>"""
    site.page("/", "Overview", "How robust are NLP measurements of changing gender representation in U.S. news, 1900–1963?", body)


def timeline(site: Site):
    rt = load("role_trends", []) or []
    comp = load("composition", []) or []
    ex = load("extraction", []) or []
    ts = load("trend_summary", []) or []
    allp = where(rt, estimand="female_share", tier="hp", population="ner")
    ref = series(allp, "all gender-signalled people (raw)", "neutral", 2, ci_on=False, width=1.5)
    figs = []
    base = where(rt, estimand="female_share", population="ner")
    ss = [series(where(base, tier="hp"), "honorific + pronoun (primary)", "women", 1),
          series(where(base, tier="h"), "honorific only", "women", 2, ci_on=False, width=1.5),
          series(where(base, tier="hpn"), "+ gendered nouns", "women", 3, ci_on=False, width=1.5)]
    figs.append(figure("line", clean_json({"title": "All people, raw", "series": ss}), "All gender-signalled people — raw shares",
                       "Three gender-evidence rules. Band: 95% CI, primary rule.", series_table(ss), SOURCE,
                       variant={"role": "ALL", "view": "raw"}))
    for role in ["AUTHORITY"] + ROLES:
        for m in ("B", "A"):
            rows = where(rt, estimand="female_share_in_role", role=role, method=m)
            s = series(rows, f"{ROLE_LABEL[role]} (method {m})", "women", 1, min_n=30)
            figs.append(figure("line", clean_json({"title": ROLE_LABEL[role], "series": [s, ref]}),
                               f"{ROLE_LABEL[role]} — women's share, method {METHOD_NAME[m]}",
                               "Years with fewer than 30 gender-signalled people in the role are omitted.",
                               series_table([s, ref]), SOURCE, variant={"role": role, "method": m, "view": "raw"}))
    adj_names = {"female_share": "ALL", "female_share_in_AUTHORITY_B": "AUTHORITY", "female_share_in_CIVIC_B": "CIVIC"}
    for oname, role in adj_names.items():
        rows = [r for r in comp if r.get("kind") == "adjusted_yearly" and r.get("outcome") == oname]
        if not rows:
            continue
        s = series(rows, "adjusted for topic + newspaper", "women", 1)
        raw = allp if role == "ALL" else where(rt, estimand="female_share_in_role", role=role, method="B")
        r2 = series(raw, "raw", "neutral", 2, ci_on=False, width=1.5, min_n=30)
        variant = {"role": role, "view": "adjusted"} if role == "ALL" else {"role": role, "method": "B", "view": "adjusted"}
        figs.append(figure("line", clean_json({"title": f"{ROLE_LABEL[role]} adjusted", "series": [s, r2]}),
                           f"{ROLE_LABEL[role]} — composition-adjusted vs raw",
                           "Adjusted series: the share each year would show with the pooled 1900–63 mix of topics and newspapers "
                           "(linear probability model with year, topic and newspaper effects; 95% article-cluster bootstrap CI).",
                           series_table([s, r2]), SOURCE, variant=variant))
    opts_role = "".join(f'<option value="{r}">{E(ROLE_LABEL[r])}</option>' for r in ["ALL", "AUTHORITY"] + ROLES)
    filters = f"""<div class="filters" data-filterbar="#tl-figs">
<label>Role<select data-filter="role">{opts_role}</select></label>
<label>Role method<select data-filter="method"><option value="B">B · dependency rules</option><option value="A">A · lexical window</option></select></label>
<label>View<select data-filter="view"><option value="raw">Raw shares</option><option value="adjusted">Adjusted for topic + newspaper</option></select></label>
<label>Years<select data-range><option value="1900-1963">1900–1963</option><option value="1900-1921">1900–1921</option><option value="1924-1963">1924–1963</option></select></label>
</div>"""
    unk = [{"label": lab, "color": "neutral", "dash": d, "points": [{"x": r["year"], "y": r.get(col)} for r in ex if ok(r.get(col))]}
           for d, (col, lab) in enumerate([("unknown_share_h", "honorific only"), ("unknown_share_hp", "honorific + pronoun (primary)"),
                                           ("unknown_share_hpn", "+ gendered nouns")], start=1)]
    bnd = [{"label": "women's share if every UNKNOWN were a woman", "color": "women", "dash": 2,
            "points": [{"x": r["year"], "y": r.get("female_share_upper_bound")} for r in ex]},
           {"label": "women's share of classified people", "color": "women", "dash": 1,
            "points": [{"x": r["year"], "y": r.get("female_share_classified")} for r in ex]},
           {"label": "women's share if every UNKNOWN were a man", "color": "women", "dash": 3,
            "points": [{"x": r["year"], "y": r.get("female_share_lower_bound")} for r in ex]}]
    h = first(ts, estimand="female_share", tier="hp", population="ner", role="ALL")
    rows = [[E(r.get("tier", "")), signed(r.get("slope_pp_dec")), ci(r.get("ci_lo"), r.get("ci_hi")), pct(r.get("early_share")), pct(r.get("late_share"))]
            for r in where(ts, estimand="female_share", population="ner")]
    body = f"""{head("Timeline", f"Women's share of gender-signalled people {trend_words(h.get('slope_pp_dec'), h.get('ci_lo'), h.get('ci_hi'))} over 1900–1963: {signed(h.get('slope_pp_dec'))} percentage points per decade ({ci(h.get('ci_lo'), h.get('ci_hi'))}; Newey–West errors on the yearly series). Use the filters to see a role, a method, or the composition-adjusted series.")}
{filters}
<div id="tl-figs">{''.join(figs)}<p class="viz-empty" data-empty hidden>No series for this combination. Adjusted series exist for all people, authority and civic roles under method B.</p></div>
<h2>Trend by gender-evidence rule</h2>
{table(["Rule", "Slope (pp/decade)", "95% CI", "1900–21", "1948–63"], rows, numeric={1, 3, 4})}
<h2>How many people can be gendered at all</h2>
<p class="prose">A trend in women's share is only interpretable next to the share of people the text leaves ungendered. If that share moves, so can the apparent trend.</p>
{figure("line", clean_json({"title": "UNKNOWN share", "series": unk, "y": {"domain": [0, 1], "format": "pct0"}}), "People with no textual gender signal", "Share of NER person entities left UNKNOWN under each rule.", series_table(unk), SOURCE)}
{figure("line", clean_json({"title": "Bounds", "series": bnd, "y": {"domain": [0, 1], "format": "pct0"}}), "Women's share of all people, under the extreme treatments of UNKNOWN", "The truth for the whole population lies between the upper and lower lines.", series_table(bnd), SOURCE)}"""
    site.page("/timeline/", "Timeline", "Women's share of people in U.S. newspapers, 1900–1963, raw and composition-adjusted.", body)


def roles(site: Site):
    rt = load("role_trends", []) or []
    ts = load("trend_summary", []) or []
    panels = []
    for role in ["AUTHORITY"] + ROLES:
        ss = [series(where(rt, estimand="female_share_in_role", role=role, method="B"), "B · dependency rules", "women", 1, min_n=30),
              series(where(rt, estimand="female_share_in_role", role=role, method="A"), "A · lexical window", "neutral", 2, min_n=30, ci_on=False)]
        panels.append({"title": ROLE_LABEL[role], "series": ss})
    tab_rows = []
    for p in panels:
        for s in p["series"]:
            for q in s["points"]:
                tab_rows.append([E(p["title"]), E(s["label"]), str(q["x"]), pct(q["y"]), num(q.get("n"))])
    fig1 = figure("multiples", clean_json({"panels": panels, "y": {"domain": [0, 1], "format": "pct0"}}),
                  "Women's share of gender-signalled people in each role",
                  "Solid: method B with 95% CI. Dashed: method A. Years with under 30 people in a role are omitted.",
                  table(["Role", "Method", "Year", "Women's share", "n"], tab_rows, numeric={3, 4}), SOURCE)
    panels2 = []
    for role in ["AUTHORITY"] + ROLES:
        panels2.append({"title": ROLE_LABEL[role], "series": [
            series(where(rt, estimand="role_rate_F", role=role, method="B"), "women", "women", 1, ci_on=False),
            series(where(rt, estimand="role_rate_M", role=role, method="B"), "men", "men", 2, ci_on=False)]})
    tab2 = [[E(p["title"]), E(s["label"]), str(q["x"]), pct(q["y"])] for p in panels2 for s in p["series"] for q in s["points"]]
    fig2 = figure("multiples", clean_json({"panels": panels2, "y": {"format": "pct0"}}),
                  "Share of women, and of men, placed in each role (method B)",
                  "Read across a panel: how much more often one gender is placed in the role. Y-axes are shared.",
                  table(["Role", "Gender", "Year", "Share placed in role"], tab2, numeric={3}), SOURCE)
    rows = []
    for role in ["AUTHORITY"] + ROLES:
        for m in ("B", "A"):
            r = first(ts, estimand="female_share_in_role", role=role, method=m)
            if r:
                rows.append([E(ROLE_LABEL[role]), m, signed(r.get("slope_pp_dec")), ci(r.get("ci_lo"), r.get("ci_hi")),
                             pct(r.get("early_share")), pct(r.get("late_share")), num(r.get("n_total"))])
    defs = "".join(f"<tr><td>{E(ROLE_LABEL[r])}</td><td>{E(ROLE_DEF[r])}</td></tr>" for r in ROLES)
    body = f"""{head("Roles", "Who is placed in public office, business, the professions, club life, the society page or the crime column, and how that changed. Every role here is measured two ways; the <a href='/methods/'>methods page</a> adds three more.")}
{fig1}
{fig2}
<h2>Trends by role</h2>
{table(["Role", "Method", "Slope (pp/decade)", "95% CI", "1900–21", "1948–63", "People"], rows, numeric={2, 4, 5, 6})}
<h2>The taxonomy</h2>
<p class="prose">Roles are multi-label: one person can be a club president and a senator's widow. <strong>Authority</strong> (public office ∪ business ∪ professional) was fixed before any outcome was computed. Civic and society-page roles were added after the corpus profile showed that women-signalled people concentrate there; without them most of those people would carry no role at all.</p>
<div class="table-scroll"><table><thead><tr><th>Role</th><th>Covers</th></tr></thead><tbody>{defs}</tbody></table></div>"""
    site.page("/roles/", "Roles", "Gender and role in U.S. newspapers, 1900–1963, measured by two methods.", body)


def quotes(site: Site):
    qt = load("quotes", []) or []
    val = load("validation", []) or []
    ment = series(where(qt, estimand="female_share_mentions"), "all gender-signalled people", "neutral", 2)
    spk = series(where(qt, estimand="female_share_speakers", attribution="name"), "people quoted (named speaker)", "women", 1)
    spk2 = series(where(qt, estimand="female_share_speakers", attribution="name+pronoun"), "people quoted (incl. he/she said)", "women", 3, ci_on=False, width=1.5)
    fig1 = figure("line", clean_json({"title": "Mentioned vs quoted", "series": [spk, spk2, ment], "y": {"format": "pct0"}}),
                  "Women's share among people mentioned, and among people quoted", "Bands: 95% article-cluster bootstrap CIs.",
                  series_table([spk, spk2, ment]), SOURCE)
    qr = [series(where(qt, estimand="quote_rate_F", attribution="name"), "women", "women", 1),
          series(where(qt, estimand="quote_rate_M", attribution="name"), "men", "men", 2)]
    fig2 = figure("line", clean_json({"title": "Quote rate", "series": qr, "y": {"format": "pct0"}}),
                  "Share of gender-signalled people who are quoted at least once", "Named-speaker attribution only.",
                  series_table(qr), SOURCE)
    dq = [series(where(qt, estimand="direct_share_F", attribution="name"), "women", "women", 1, ci_on=False),
          series(where(qt, estimand="direct_share_M", attribution="name"), "men", "men", 2, ci_on=False)]
    fig3 = figure("line", clean_json({"title": "Direct quotation share", "series": dq, "y": {"format": "pct0"}}),
                  "Of quoted people, the share quoted directly (in quotation marks)", "", series_table(dq), SOURCE)
    h7 = [r for r in qt if r.get("estimand") == "H7_speaker_minus_mention_share"]
    rows = [[E(r.get("attribution", "")), E(r.get("period", "")), signed(r.get("diff_pp")), ci(r.get("ci_lo"), r.get("ci_hi")),
             num(r.get("n_speakers")), num(r.get("n_mentions"))] for r in h7]
    vq = [[E(r["method"]), dec(r.get("precision")), dec(r.get("recall")), dec(r.get("f1")), num(r.get("n_pos_ref"))]
          for r in val if r.get("target") == "quoted"]
    pooled = next((r for r in h7 if r.get("attribution") == "name" and r.get("period") == "all"), {})
    body = f"""{head("Quotes", f"Among people quoted by name, women's share differs from their share of all gender-signalled people by {signed(pooled.get('diff_pp'))} percentage points over the whole period ({ci(pooled.get('ci_lo'), pooled.get('ci_hi'))}).")}
{fig1}
<h2>Speaker share minus mention share</h2>
{table(["Attribution", "Period", "Difference (pp)", "95% CI", "Speakers", "People"], rows, numeric={2, 4, 5})}
{fig2}
{fig3}
<h2>What counts as a quotation</h2>
<div class="prose"><p>A <strong>speech event</strong> is a verb from a fixed list (say, tell, declare, announce, testify, …) whose grammatical subject is a person mention. Quotation marks alone never count without an attributed speaker. A quotation is <em>direct</em> if the sentence contains quotation marks, or if a short attribution fragment follows a quoted sentence. The secondary rule also credits “he said” and “she said” to the nearest preceding salient person of the matching signalled gender.</p></div>
<h2>How well the rule finds quoted people</h2>
{table(["Rule", "Precision", "Recall", "F1", "Quoted people in reference sample"], vq, numeric={1, 2, 3, 4})}"""
    site.page("/quotes/", "Quotes", "Who is quoted in U.S. newspapers, 1900–1963, compared with who is mentioned.", body)


def language(site: Site):
    lg = load("language", {}) or {}
    ag = lg.get("agency", [])
    ss = [series([r for r in ag if r.get("kind") == "agency_yearly" and r.get("gender") == "F"], "women", "women", 1),
          series([r for r in ag if r.get("kind") == "agency_yearly" and r.get("gender") == "M"], "men", "men", 2)]
    fig = figure("line", clean_json({"title": "Agency", "series": ss, "y": {"format": "pct0"}}),
                 "Share of a person's syntactic positions that are agent rather than patient",
                 "Agent: subject of a non-copular verb. Patient: passive subject or direct object. Band: 95% CI.",
                 series_table(ss), SOURCE)
    lo = lg.get("logodds", [])
    fams = [("agent_verb", "Verbs with the person as agent"), ("patient_verb", "Verbs with the person as patient"),
            ("modifier", "Adjectives modifying the person"), ("possessed_noun", "Nouns the person possesses")]
    periods = ["all", "1900-15", "1918-30", "1933-45", "1948-63"]
    blocks = []
    for per in periods:
        parts = []
        for fam, title in fams:
            rows = sorted([r for r in lo if r.get("family") == fam and r.get("period") == per], key=lambda r: r.get("z") or 0)
            if not rows:
                continue
            def fmt_rows(rs):
                return [[E(r["word"]), num(r.get("n_F")), num(r.get("n_M")), dec(r.get("z")),
                         ("q < .05" if ok(r.get("q")) and r["q"] < 0.05 else "n.s.")] for r in rs]
            parts.append(f"<h3>{E(title)}</h3><div class='two-col'><div class='table-scroll'>"
                         f"{table(['More associated with women', 'n women', 'n men', 'z', 'FDR'], fmt_rows(rows[::-1][:15]), numeric={1, 2, 3})}"
                         f"</div><div class='table-scroll'>{table(['More associated with men', 'n women', 'n men', 'z', 'FDR'], fmt_rows(rows[:15]), numeric={1, 2, 3})}</div></div>")
        blocks.append(f'<section data-variant="{E(json.dumps({"period": per}))}">{"".join(parts)}</section>')
    opts = "".join(f'<option value="{p}">{"1900–1963 pooled" if p == "all" else p.replace("-", "–")}</option>' for p in periods)
    tr = {r["gender"]: r for r in ag if r.get("kind") == "agency_trend"}
    body = f"""{head("Language", "Which actions and descriptions attach to people signalled as women and as men. Everything on this page is exploratory: many words are tested, and differences are reported with false-discovery-rate control, not as confirmed findings.")}
{fig}
<p class="prose">Agency trend: women {signed((tr.get('F') or {}).get('slope_pp_dec'))} pp per decade ({ci((tr.get('F') or {}).get('ci_lo'), (tr.get('F') or {}).get('ci_hi'))}); men {signed((tr.get('M') or {}).get('slope_pp_dec'))} pp per decade ({ci((tr.get('M') or {}).get('ci_lo'), (tr.get('M') or {}).get('ci_hi'))}).</p>
<h2>Distinctive words</h2>
<p class="prose">Weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi &amp; Quinn 2008), computed <em>within</em> each coarse section and pooled by inverse variance, so a word does not look “female” merely because it is common on the society page. Benjamini–Hochberg FDR at q = .05 across each word family.</p>
<div class="filters" data-filterbar="#lang-tabs"><label>Period<select data-filter="period">{opts}</select></label></div>
<div id="lang-tabs">{''.join(blocks)}</div>"""
    site.page("/language/", "Language", "Agency verbs and descriptive language by signalled gender, 1900–1963 (exploratory).", body)


def topics(site: Site):
    comp = load("composition", []) or []
    tm = load("topic_model", {}) or {}
    gp = [r for r in comp if r.get("kind") == "group_period" and r.get("group") == "topic_label"]
    periods = ["1900-15", "1918-30", "1933-45", "1948-63"]
    tl = sorted({r["group_value"] for r in gp})
    share = {(r["group_value"], r["period"]): r for r in gp}
    order = sorted(tl, key=lambda t: -sum((share.get((t, p)) or {}).get("female_share") or 0 for p in periods))
    cells = [[(share.get((t, p)) or {}).get("female_share") for p in periods] for t in order]
    counts = [[(share.get((t, p)) or {}).get("n") for p in periods] for t in order]
    vmax = max([c for row in cells for c in row if ok(c)] or [1])
    hm = figure("heatmap", clean_json({"rows": order, "cols": [p.replace("-", "–") for p in periods], "cells": cells, "counts": counts,
                                       "format": "pct0", "vmin": 0, "vmax": vmax, "valueLabel": "women's share", "title": "Topic by period"}),
                "Women's share of gender-signalled people, by topic and period", "Topics ordered by average share. Cells print their value.",
                table(["Topic"] + periods, [[E(t)] + [f"{pct(c, 0)} (n={num(n)})" for c, n in zip(cr, nr)] for t, cr, nr in zip(order, cells, counts)], numeric={1, 2, 3, 4}), SOURCE)
    mix = [[(share.get((t, p)) or {}).get("entity_share_of_period") for p in periods] for t in order]
    vmax2 = max([c for row in mix for c in row if ok(c)] or [1])
    hm2 = figure("heatmap", clean_json({"rows": order, "cols": [p.replace("-", "–") for p in periods], "cells": mix, "format": "pct0",
                                        "vmin": 0, "vmax": vmax2, "valueLabel": "share of gender-signalled people", "title": "Topic mix"}),
                 "Where gender-signalled people appear: topic mix by period", "Each column sums to 100%.",
                 table(["Topic"] + periods, [[E(t)] + [pct(c, 1) for c in r] for t, r in zip(order, mix)], numeric={1, 2, 3, 4}), SOURCE)
    dc = [r for r in comp if r.get("kind") == "decomposition"]
    glab = {"topic_label": "topic", "section": "section", "paper_group": "Evening Star vs other papers"}
    figs_dc = []
    for oname, title in (("female_share", "All gender-signalled people"), ("female_share_in_AUTHORITY_B", "Authority roles (method B)")):
        rows = []
        for r in [x for x in dc if x.get("outcome") == oname]:
            for part, col in (("total change", "total"), ("composition", "composition"), ("within-group", "within")):
                rows.append({"label": f"{glab.get(r['group'], r['group'])}: {part}", "value": r.get(f"{col}_pp"),
                             "lo": r.get(f"{col}_lo"), "hi": r.get(f"{col}_hi"), "color": "women" if col == "within" else "neutral"})
        figs_dc.append(figure("bars", clean_json({"rows": rows, "format": "pp", "title": title}),
                              f"{title}: change 1900–21 → 1948–63, split into composition and within-group parts",
                              "Symmetric Kitagawa decomposition; whiskers are 95% article-cluster bootstrap CIs.",
                              table(["Component", "pp", "95% CI"], [[E(r["label"]), signed(r["value"]), ci(r["lo"], r["hi"])] for r in rows], numeric={1}), SOURCE))
    lp = [r for r in comp if r.get("kind") == "lpm"]
    fe_lab = {"none": "raw", "topic": "topic FE", "paper": "newspaper FE", "topic+paper": "topic + newspaper FE", "topic+paper+page": "+ page position"}
    samp_lab = {"all": "all newspapers", "excl_evening_star": "excluding Evening Star", "news_only": "news only (no ads, notices, fiction)",
                "pre_1922": "1900–1921 only", "post_1923": "1924–1963 only"}
    dots = []
    for oname, title in (("female_share", "All gender-signalled people"), ("female_share_in_AUTHORITY_B", "Authority roles (method B)")):
        rows = [{"label": f"{fe_lab.get(r['fe'], r['fe'])} · {samp_lab.get(r['sample'], r['sample'])}", "value": r.get("slope_pp_dec"),
                 "lo": r.get("ci_lo"), "hi": r.get("ci_hi"), "hollow": r.get("sample") != "all"}
                for r in lp if r.get("outcome") == oname and r.get("sample") in ("all", "excl_evening_star", "news_only")]
        dots.append(figure("dots", clean_json({"rows": rows, "zero": True, "format": "dec2", "labelWidth": 300,
                                               "xLabel": "pp per decade", "title": title, "color": "women"}),
                           f"{title}: trend under different composition controls",
                           "Linear probability model; standard errors clustered two ways (newspaper, year). Filled: all newspapers.",
                           table(["Specification", "Slope (pp/decade)", "95% CI"], [[E(r["label"]), signed(r["value"]), ci(r["lo"], r["hi"])] for r in rows], numeric={1}), SOURCE))
    top = tm.get("top_terms", {})
    stab = tm.get("stability_matched_cosine_mean", {})
    labels = load("topic_labels", {}) or {}
    trows = [[str(k), E(labels.get("topics", {}).get(str(k), {}).get("label", "—")), dec(stab.get(str(k))), E(" ".join(top.get(str(k), [])[:10]))]
             for k in sorted(top, key=int)]
    agree = tm.get("assignment_agreement_across_seeds", [])
    body = f"""{head("Topics", "Newspapers changed what they printed, and the digitised corpus changed which newspapers it holds. This page separates the two from any change in how people were written about.")}
{hm}
{hm2}
<h2>Composition or change?</h2>
{''.join(figs_dc)}
{''.join(dots)}
<h2>The topic model</h2>
<p class="prose">NMF with 24 components on TF-IDF of article text, fitted on a year-balanced subsample of {num(tm.get('fit_articles'))} articles and three random seeds. Components are matched across seeds by Hungarian assignment on topic–term cosine. Mean matched cosine is {dec(tm.get('stability_matched_cosine_overall'))}; article-level assignment agreement across seeds is {', '.join(dec(a) for a in agree) or '—'}. Labels were assigned by reading top terms, and they are interpretive. Unstable topics (cosine below .75) are flagged in <code>config/topic_labels.json</code>, and the coarse section grouping is the more robust unit.</p>
<div class="table-scroll">{table(["#", "Label", "Seed stability", "Top terms"], trows, numeric={2})}</div>"""
    site.page("/topics/", "Topics", "Topic and newspaper composition, and trends adjusted for them.", body)


def methods(site: Site):
    ma = load("methods", []) or []
    rel = site.rel
    yr = where(ma, kind="yearly", role="AUTHORITY")
    ms = [m for m in ("A", "B", "C", "D", "E") if where(yr, method=m)]
    panels = [{"title": METHOD_NAME[m], "series": [series([r for r in where(yr, method=m)], METHOD_NAME[m], "women", 1, min_n=15, ci_on=False)]} for m in ms]
    tab = [[E(METHOD_NAME[m]), str(r["year"]), pct(r.get("share")), pct(r.get("prevalence")), num(r.get("n"))]
           for m in ms for r in sorted(where(yr, method=m), key=lambda r: r["year"])]
    fig1 = figure("multiples", clean_json({"panels": panels, "y": {"format": "pct0"}}),
                  "Women's share of authority roles, measured five ways on the same people",
                  "Each panel is one method applied to the same application sample (up to 5,000 gender-signalled people per year; the LLM saw a 300-per-year subsample).",
                  table(["Method", "Year", "Women's share in role", "Role prevalence", "People in role"], tab, numeric={2, 3, 4}), SOURCE)
    tr = where(ma, kind="trend", role="AUTHORITY")
    rows = [{"label": METHOD_NAME[r["method"]], "value": r.get("slope_pp_dec"), "lo": r.get("ci_lo"), "hi": r.get("ci_hi")}
            for r in tr if r.get("method") in METHOD_NAME]
    fig2 = figure("dots", clean_json({"rows": rows, "zero": True, "format": "dec2", "xLabel": "pp per decade", "title": "Slopes", "color": "women"}),
                  "Trend in women's share of authority roles, by method", "HAC 95% CIs on the yearly series.",
                  table(["Method", "Slope (pp/decade)", "95% CI"], [[E(r["label"]), signed(r["value"]), ci(r["lo"], r["hi"])] for r in rows], numeric={1}), SOURCE)
    h4 = where(ma, kind="h4")
    h4rows = [[E(ROLE_LABEL.get(r["role"], r["role"])), signed(r.get("slope_min")), signed(r.get("slope_max")), dec(r.get("abs_ratio_max_min")),
               "yes" if r.get("signs_agree") else "no", E(r.get("methods", ""))] for r in h4]
    kap = [r for r in ma if r.get("kind") == "kappa" and r.get("role") == "AUTHORITY" and r.get("period") != "all"]
    pairs = sorted({f"{r['method_a']}–{r['method_b']}" for r in kap})
    pers = ["1900-15", "1918-30", "1933-45", "1948-63"]
    kd = {(f"{r['method_a']}–{r['method_b']}", r["period"]): r.get("kappa") for r in kap}
    kcells = [[kd.get((p, q)) for q in pers] for p in pairs]
    fig3 = figure("heatmap", clean_json({"rows": pairs, "cols": [p.replace("-", "–") for p in pers], "cells": kcells, "format": "dec2",
                                         "vmin": 0, "vmax": 1, "valueLabel": "Cohen's κ", "title": "Agreement"}),
                  "Agreement between methods on who holds an authority role (Cohen's κ)", "Per period, on the shared application sample.",
                  table(["Pair"] + pers, [[E(p)] + [dec(c) for c in row] for p, row in zip(pairs, kcells)], numeric={1, 2, 3, 4}), SOURCE)
    kt = [[E(f"{r['method_a']}–{r['method_b']}"), E(ROLE_LABEL.get(r["role"], r["role"])), dec(r.get("kappa_slope_per_decade"), 3),
           f"{dec(r.get('ci_lo'), 3)} to {dec(r.get('ci_hi'), 3)}"] for r in ma if r.get("kind") == "kappa_trend" and r.get("role") in ("AUTHORITY", "CIVIC", "PUBLIC_OFFICE")]
    llm = (rel.get("models", {}) or {}).get("llm_E", {})
    llm_rows = [[E(k), num(v.get("n")), num(v.get("n_parse_ok")), E(", ".join(v.get("models", []))), E(", ".join(v.get("prompt_hashes", []))),
                 E(f"{v.get('first_ts', '')[:10]} to {v.get('last_ts', '')[:10]}")] for k, v in llm.items()]
    defs = "".join(f"<tr><td>{E(METHOD_NAME[m])}</td><td>{E(METHOD_DEF[m])}</td></tr>" for m in METHOD_NAME)
    body = f"""{head("Methods", "The same people, the same years, five ways of deciding whether someone holds a role. If the historical story depended on the method, it would show here.")}
<div class="table-scroll"><table><thead><tr><th>Method</th><th>How it decides</th></tr></thead><tbody>{defs}</tbody></table></div>
{fig1}
{fig2}
<h2>How far apart are the methods? (H4)</h2>
<p class="prose">For each role, the spread between the smallest and largest method slope on the same population. The pre-registered test counts a role as materially different when the ratio of the largest to the smallest absolute slope exceeds 2, or when the signs disagree.</p>
<div class="table-scroll">{table(["Role", "Smallest slope", "Largest slope", "Ratio", "Signs agree", "Methods"], h4rows, numeric={1, 2, 3})}</div>
{fig3}
<h3>Does agreement change over time?</h3>
<p class="prose">A trend in κ means the methods disagree more in some decades than others, so disagreement is itself time-dependent.</p>
<div class="table-scroll">{table(["Pair", "Role", "κ change per decade", "95% CI"], kt, numeric={2})}</div>
<h2>The LLM condition, for the record</h2>
<div class="table-scroll">{table(["Output file", "Calls", "Parsed", "Model", "Prompt hash", "Dates"], llm_rows, numeric={1, 2})}</div>"""
    site.page("/methods/", "Methods", "Five NLP methods for measuring roles, compared on the same people.", body)


def mark_excerpt(text, s, e):
    s, e = max(0, s), min(len(text), e)
    return f"{E(text[:s])}<mark>{E(text[s:e])}</mark>{E(text[e:])}"


def disagreement(site: Site):
    ex = (load("disagreement_examples", {}) or {}).get("examples", [])
    by_role = {}
    for x in ex:
        by_role.setdefault(x["role"], []).append(x)
    blocks = []
    for role, xs in by_role.items():
        items = []
        for x in xs:
            labs = "".join(f'<li class="{"yes" if v else "no"}">{E(METHOD_NAME.get(k, k))}: {"yes" if v else "no"}'
                           f'{(" (" + dec(x["probs"].get(k), 2) + ")") if k in x.get("probs", {}) else ""}</li>' for k, v in x["labels"].items())
            items.append(f'<li><p class="meta">{x["year"]} · {E(x["publication"])} · gender signal: {E(x["gender_signal"])} · {E(x["entity_id"])}</p>'
                         f'<blockquote>{mark_excerpt(x["excerpt"], x["hl_start"], x["hl_end"])}</blockquote><ul class="labels">{labs}</ul></li>')
        blocks.append(f'<section data-variant="{E(json.dumps({"role": role}))}"><h2>{E(ROLE_LABEL.get(role, role))}</h2><ul class="evidence">{"".join(items)}</ul></section>')
    opts = "".join(f'<option value="{r}">{E(ROLE_LABEL.get(r, r))}</option>' for r in by_role)
    body = f"""{head("Disagreement", "Passages where the role methods reach different verdicts about the same person. The highlighted span is the person; each method's verdict is listed beneath, with the classifier probability where there is one.")}
<p class="content-note">These are unedited excerpts of public-domain newspaper OCR from 1900–1963. They contain OCR errors and may contain racist, sexist or violent historical language. They are shown because a label is only credible if its evidence can be read.</p>
<div class="filters" data-filterbar="#dis"><label>Role<select data-filter="role">{opts}</select></label></div>
<div id="dis">{''.join(blocks) or '<p class="viz-empty">No disagreement examples were exported.</p>'}</div>"""
    site.page("/disagreement/", "Disagreement", "Examples where NLP role-measurement methods disagree, with the source text.", body)


def validation(site: Site):
    val = load("validation", []) or []
    gc = load("gender_consistency", []) or []
    n_ref = next((r.get("n_reference") for r in val if ok(r.get("n_reference"))), None)
    ref_note = next((r.get("reference") for r in val if r.get("reference")), "")
    isp = first(val, target="is_person")
    g = [[E(r["method"]), dec(r.get("precision")), dec(r.get("recall")), pct(r.get("prev_pred_w"), 0), num(r.get("n_pred_pos"))]
         for r in val if r.get("target") == "gender"]
    gfm = [[E(r["method"]), E(r["target"].replace("gender_", "")), dec(r.get("precision")), dec(r.get("recall")), dec(r.get("f1"))]
           for r in val if r.get("target") in ("gender_F", "gender_M")]
    wo = [[E(r["method"]), pct(r.get("precision")), num(r.get("n_pred_pos")), num(r.get("n_pos_ref"))]
          for r in val if r.get("target") == "gender_assigned_without_text_signal"]
    methods = [m for m in ("A_lexical", "B_dependency", "C_bow@0.5", "D_embed@0.5", "E_llm") if any(r.get("method") == m for r in val)]
    targets = ["AUTHORITY"] + ROLES
    f1 = {(r["method"], r["target"]): r.get("f1") for r in val}
    cells = [[f1.get((m, t)) for t in targets] for m in methods]
    hm = figure("heatmap", clean_json({"rows": methods, "cols": [ROLE_LABEL[t].split(" (")[0] for t in targets], "cells": cells, "format": "dec2",
                                       "vmin": 0, "vmax": 1, "valueLabel": "F1", "labelWidth": 120, "title": "Role F1"}),
                "Role F1 against reference labels, by method", "Sampling-weighted to the population. Blank: no positives in the reference sample.",
                table(["Method"] + targets, [[E(m)] + [dec(c) for c in row] for m, row in zip(methods, cells)], numeric=set(range(1, len(targets) + 1))), "")
    prf = [[E(r["method"]), E(r["target"]), dec(r.get("precision")), dec(r.get("recall")), dec(r.get("f1")), num(r.get("n_pos_ref")),
            pct(r.get("prev_ref_w")), pct(r.get("prev_pred_w"))]
           for r in val if r.get("target") in ("AUTHORITY", "PUBLIC_OFFICE", "PROFESSIONAL", "BUSINESS", "CIVIC", "FAMILY")]
    ac = load("annotation_consistency", {}) or {}
    ac_rows = [[E(r["field"].replace("role:", "role · ")), dec(r.get("kappa")), pct(r.get("agreement"), 0), num(r.get("n"))]
               for r in ac.get("rows", [])]
    ac_html = (f"<h3>Self-consistency of the reference labels</h3><p class='prose'>{num(ac.get('n_items'))} items were re-labelled blind, "
               f"in shuffled order, after the full pass. This is {E(ac.get('what', ''))}. It measures stability, not correctness, "
               f"and it is not agreement between annotators.</p><div class='table-scroll'>"
               f"{table(['Field', 'Cohen κ', 'Agreement', 'Items'], ac_rows, numeric={1, 2, 3})}</div>") if ac_rows else ""
    tot = next((r for r in gc if r.get("year") == -1), {})
    cons = [{"label": "pronoun rule agrees with honorific", "color": "neutral", "dash": 1,
             "points": [{"x": r["year"], "y": r.get("agree"), "n": r.get("n")} for r in gc if r.get("year", -1) > 0]}]
    body = f"""{head("Validation", "Every method is checked against a stratified sample of reference labels. The reference labels are not human annotation, and this page says so first.")}
<div class="finding"><p class="claim">Who produced the reference labels</p>
<p>{E(ref_note)}. The {num(n_ref)} labelled people were drawn by stratified random sampling (period × honorific class × role present). The annotator worked blind to every method's output, following <code>research/annotation_protocol.md</code>. Estimates are reweighted to population proportions. Replacing these labels with human annotation is the first open item, and the <a href="/annotate/">annotation tool</a> exists for exactly that.</p></div>
{ac_html}
<h2>Is the extracted span a person?</h2>
<p class="prose">Of sampled entities, {pct(isp.get('precision'))} (sampling-weighted) are real references to a person. The rest are OCR fragments, places and organisations tagged as people. Every entity-level estimate carries this noise.</p>
<h2>Gender signal</h2>
<p class="prose">For each gender-evidence rule: <em>accuracy when it assigns a gender</em>, and <em>recall of the genders the text does signal</em>.</p>
<div class="table-scroll">{table(["Rule", "Accuracy when assigning", "Recall of signalled genders", "Share assigned", "Assigned (n)"], g, numeric={1, 2, 3, 4})}</div>
<div class="table-scroll">{table(["Rule", "Gender", "Precision", "Recall", "F1"], gfm, numeric={2, 3, 4})}</div>
<h3>Genders assigned where the text gives none</h3>
<p class="prose">This is the failure mode an LLM is most prone to: guessing gender from a first name or from occupational stereotype when the text is silent.</p>
<div class="table-scroll">{table(["Method", "Share of text-silent people given a gender", "Given a gender (n)", "Text-silent people (n)"], wo, numeric={1, 2, 3})}</div>
{figure("line", clean_json({"title": "Pronoun rule vs honorific", "series": cons, "y": {"domain": [0, 1], "format": "pct0"}}), "Large-sample check: does the pronoun rule agree with the honorific?", f"Among {num(tot.get('n'))} people with both an honorific and a pronoun signal, the rule agrees {pct(tot.get('agree'))} of the time (women {pct(tot.get('agree_when_F'))}, men {pct(tot.get('agree_when_M'))}). No annotation is needed: the honorific is the check.", series_table(cons), SOURCE)}
<h2>Roles</h2>
{hm}
<div class="table-scroll">{table(["Method", "Role", "Precision", "Recall", "F1", "Reference positives", "Reference prevalence", "Predicted prevalence"], prf, numeric={2, 3, 4, 5, 6, 7})}</div>"""
    site.page("/validation/", "Validation", "Precision, recall and F1 of every NLP method against stratified reference labels.", body)


def robustness(site: Site):
    cells = load("robustness_cells", []) or []
    summ = load("robustness_summary", []) or []
    groups = [("H1_female_share", "-", "H1 · all people"), ("H2_female_share_in_AUTHORITY", "A", "H2 authority · A"),
              ("H2_female_share_in_AUTHORITY", "B", "H2 authority · B"), ("H2_female_share_in_AUTHORITY", "A(app)", "H2 authority · A (app. sample)"),
              ("H2_female_share_in_AUTHORITY", "B(app)", "H2 authority · B (app. sample)")]
    for pre in ("C", "D"):
        for th in ("0.3", "0.5", "0.7"):
            groups.append(("H2_female_share_in_AUTHORITY", f"{pre}@{th}", f"H2 authority · {pre} at {th}"))
    groups.append(("female_share_in_CIVIC", "B", "Civic roles · B"))
    rows = []
    for est, m, lab in groups:
        vals = [c["slope_pp_dec"] for c in cells if c.get("estimand") == est and c.get("method") == m and ok(c.get("slope_pp_dec"))]
        if vals:
            rows.append({"label": lab, "values": vals})
    fig = figure("strip", clean_json({"rows": rows, "xLabel": "trend estimate, pp per decade", "format": "dec2", "labelWidth": 240, "title": "Multiverse"}),
                 "Every specification's trend estimate", "One dot per specification; tick marks the median. The right margin gives the share of specifications with a positive trend.",
                 table(["Estimand", "Specifications", "Positive", "Median", "Min", "Max"],
                       [[E(r["label"]), num(len(r["values"])), pct(sum(v > 0 for v in r["values"]) / len(r["values"]), 0),
                         dec(sorted(r["values"])[len(r["values"]) // 2]), dec(min(r["values"])), dec(max(r["values"]))] for r in rows], numeric={1, 2, 3, 4, 5}), "")
    def dim_table(dim):
        rs = [r for r in summ if r.get(dim) is not None and all(r.get(k) is None for k in ("adjust", "tier", "papers", "method") if k != dim)]
        return table([dim.capitalize(), "Estimand", "Cells", "Positive", "Sig. positive", "Sig. negative", "Median slope", "Sign survives all"],
                     [[E(str(r.get(dim))), E(r["estimand"]), num(r.get("n_cells")), pct(r.get("share_positive"), 0), pct(r.get("share_sig_positive"), 0),
                       pct(r.get("share_sig_negative"), 0), dec(r.get("slope_median")), "yes" if r.get("sign_survives_all") else "no"] for r in rs],
                     numeric={2, 3, 4, 5, 6})
    body = f"""{head("Robustness", "Each headline estimate is re-computed under every combination of reasonable analysis choices. A claim is reported as supported only if its sign survives all of them.")}
{fig}
<h2>The grid</h2>
<div class="prose"><p><strong>Gender rule</strong>: honorific only; + pronoun rule (primary); + gendered nouns. <strong>People</strong>: NER-detected only (primary); + honorific-pattern detector. <strong>Duplicates</strong>: keep; drop near-duplicates (MinHash Jaccard ≥ .8). <strong>OCR</strong>: all; legible articles with dictionary rate ≥ .75. <strong>Newspapers</strong>: all; excluding the <em>Evening Star</em>; capped at 500 articles per paper-year. <strong>Time bins</strong>: year; 5-year; decade. <strong>Content</strong>: all; news only. <strong>Role method</strong> and classifier <strong>threshold</strong> (0.3/0.5/0.7) for role estimands. A reduced grid adds topic + newspaper fixed effects.</p></div>
<h3>By gender rule</h3><div class="table-scroll">{dim_table('tier')}</div>
<h3>By newspaper handling</h3><div class="table-scroll">{dim_table('papers')}</div>
<h3>By method</h3><div class="table-scroll">{dim_table('method')}</div>
<h3>Raw vs fixed-effects adjusted</h3><div class="table-scroll">{dim_table('adjust')}</div>"""
    site.page("/robustness/", "Robustness", "Multiverse analysis of every headline trend in GenderNews Atlas.", body)


def failures(site: Site):
    val = load("validation", []) or []
    ex = load("extraction", []) or []
    gc = load("gender_consistency", []) or []
    corpus = load("corpus", []) or []
    fx = (load("failure_examples", {}) or {}).get("examples", [])
    isp = first(val, target="is_person")
    tot = next((r for r in gc if r.get("year") == -1), {})
    q = first(val, method="quote_rule_name", target="quoted")
    wo = first(val, method="E_llm", target="gender_assigned_without_text_signal")
    conflict = [{"label": "entity gender conflicts (dropped to UNKNOWN)", "color": "neutral", "dash": 1,
                 "points": [{"x": r["year"], "y": r.get("share_src_conflict")} for r in ex]}]
    amb = [[str(r["year"]), dec(r.get("ambiguous_dropped_per_article")), dec(r.get("mentions_per_entity")), num(r.get("pattern_only_entities"))] for r in ex]
    ocr = [{"label": "median dictionary-word rate", "color": "neutral", "dash": 1, "points": [{"x": r["year"], "y": r.get("dict_rate_median")} for r in corpus]},
           {"label": "articles labelled legible", "color": "neutral", "dash": 2, "points": [{"x": r["year"], "y": r.get("share_legible")} for r in corpus]}]
    items = "".join(f'<li><p class="meta">{E(x.get("kind", ""))} · {x.get("year", "")} · {E(x.get("publication", ""))}</p>'
                    f'<blockquote>{mark_excerpt(x["excerpt"], x["hl_start"], x["hl_end"])}</blockquote><p>{E(x.get("note", ""))}</p></li>' for x in fx)
    body = f"""{head("Failures", "Where the pipeline breaks, how often, and in which direction the error pushes the estimates.")}
<h2>Named-entity recognition on OCR</h2>
<p class="prose">{pct(1 - isp['precision'] if ok(isp.get('precision')) else None)} of extracted “people” in the reference sample are not people: OCR fragments, streets, ships and firms. The error is not gender-neutral in effect. Honorific-bearing spans are almost always real people, so the gender-signalled subset is cleaner than the UNKNOWN pool.</p>
<h2>Pronouns and coreference</h2>
<p class="prose">The pronoun rule disagrees with the honorific on {pct(1 - tot['agree'] if ok(tot.get('agree')) else None)} of the {num(tot.get('n'))} people who carry both signals. The typical failure is a pronoun that refers to a second person the parser did not tag (“Smith told his wife she…”). Conflicting evidence sends a person to UNKNOWN rather than to a guess.</p>
{figure("line", clean_json({"title": "Conflicts", "series": conflict, "y": {"format": "pct"}}), "Share of people whose gender evidence conflicts", "", series_table(conflict, pct), SOURCE)}
<h2>Entity resolution</h2>
<p class="prose">Surname-only mentions that could belong to more than one person in the article are dropped rather than guessed. Period naming conventions make this frequent: a wife named by her husband's full name, “Mr. and Mrs.” pairs.</p>
<div class="table-scroll">{table(["Year", "Ambiguous mentions dropped per article", "Mentions per person", "Pattern-only people (excluded from primary)"], amb, numeric={1, 2, 3})}</div>
<h2>OCR quality</h2>
{figure("line", clean_json({"title": "OCR", "series": ocr, "y": {"domain": [0.5, 1], "format": "pct0"}}), "OCR quality proxies by year", "A period-dictionary word rate (Webster's Second, 1934) and AmericanStories' own legibility label.", series_table(ocr), SOURCE)}
<h2>Quotation attribution</h2>
<p class="prose">The named-speaker rule finds {pct(q.get('recall'))} of reference-labelled quoted people at a precision of {pct(q.get('precision'))}. Most misses are quotations attributed through a title or a pronoun the parser attached elsewhere.</p>
<h2>Historical language</h2>
<div class="prose"><p>Some failures are not errors in the usual sense but shifts in the language itself. <em>Chairman</em>, <em>alderman</em> and <em>congressman</em> were applied to women office-holders, so they are treated as role terms, never as gender evidence. Married women were routinely named by their husbands' names (“Mrs. John Smith”), which is why entity resolution is keyed to honorific class. <em>Miss</em> appears in advertising as a size category (“misses' dresses”); requiring an NER person span keeps most of these out. <em>Secretary</em> and <em>president</em> name both government offices and club officers, which is why method B types organisational heads by the organisation they head.</p></div>
<h2>LLM disagreement</h2>
<p class="prose">Where the text gives no gender signal, the LLM still assigns a gender to {pct(wo.get('precision'))} of such people in the reference sample. That is the clearest single reason its gender output is not used as evidence here.</p>
{f'<h2>Examples from the reference sample</h2><ul class="evidence">{items}</ul>' if items else ''}"""
    site.page("/failures/", "Failures", "Failure analysis: NER, coreference, OCR, quotation attribution and LLM errors.", body)


def data_page(site: Site):
    corpus = load("corpus", []) or []
    sa = load("sampling_audit", {}) or {}
    cfg = json.loads((ROOT / "config" / "corpus.json").read_text())
    ds = cfg["dataset"]
    rows = [[str(r["year"]), num(r.get("n_articles")), num(r.get("n_publications")), pct(r.get("evening_star_share")),
             num(r.get("median_words")), pct(r.get("dict_rate_median")), pct(r.get("share_legible")), pct(r.get("near_dup_share"), 2)]
            for r in corpus]
    full = sa.get("full_year_1963", {})
    audit = [[E(k), dec(v.get("tvd_observed"), 3), dec(v.get("tvd_null_median"), 3), dec(v.get("tvd_null_p95"), 3), dec(v.get("p_value_one_sided"), 2)]
             for k, v in (full.get("comparison_prefix_vs_full") or {}).items()]
    files = "".join(f'<li><a href="/data/{p.name}">{E(p.name)}</a></li>' for p in sorted(DATA.glob("*.json")))
    body = f"""{head("Data", "What the corpus is, how it was sampled, what was checked, and how to rebuild everything.")}
<section class="two-col"><div class="prose">
<h2>Corpus</h2>
<p><strong>{E(ds['name'])}</strong> ({E(ds['citation'])}). Source collection: {E(ds['source_collection'])}. Source rights: {E(ds['source_rights'])}. Dataset licence: {E(ds['license'])}.</p>
<p>AmericanStories is article-segmented OCR of digitised U.S. newspapers. It is <em>not</em> a representative sample of American news. Coverage depends on which papers libraries chose to digitise. The corpus thins sharply after 1922, when copyright begins to bind, and the Washington <em>Evening Star</em> dominates later years. Both facts are modelled, not ignored.</p>
<h2>Sampling</h2>
<p>For each of 22 years (1900–1963, every third year) the pipeline streams the year's archive and keeps the first 10,000 articles that pass length and legibility filters. The archive's member order is date-shuffled, so a streamed prefix should behave like a random sample. That was tested: against a complete download of 1963 ({num(full.get('n_scans_full_year'))} scans), the prefix's distribution over newspapers, months and page positions is indistinguishable from random samples of the same size.</p>
</div><aside class="aside"><h3>Rebuild</h3><pre><code>make setup
make data preprocess annotate
make models analyze robustness
make figures freeze paper site
make test</code></pre><p><code>make smoke</code> runs a small end-to-end check in CI.</p><h3>Downloads</h3><ul>{files}</ul></aside></section>
<h2>Sampling audit (1963, full year)</h2>
<div class="table-scroll">{table(["Dimension", "Observed TVD", "Null median", "Null 95th pct.", "p (one-sided)"], audit, numeric={1, 2, 3, 4})}</div>
<h2>Corpus by year</h2>
<div class="table-scroll">{table(["Year", "Articles", "Newspapers", "Evening Star share", "Median words", "Dictionary rate", "Legible", "Near-duplicates"], rows, numeric={1, 2, 3, 4, 5, 6, 7})}</div>
<h2>What is public and what is regenerated</h2>
<p class="prose">The repository holds code, configuration, manifests with per-year content hashes, reference labels and derived results. The sampled article text (public domain, CC-BY-4.0 as packaged) is regenerated by <code>make data</code> rather than committed, to keep the repository small. Excerpts on this site are short and shown as evidence.</p>"""
    site.page("/data/", "Data", "Corpus provenance, sampling audit and reproduction for GenderNews Atlas.", body)


# ----------------------------------------------------------------------------- paper
def md_inline(s):
    s = E(s)
    s = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f'<img alt="{m.group(1)}" src="{m.group(2).replace("../figures/", "/figures/")}">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    return s


def markdown(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append(f"<pre><code>{E(chr(10).join(buf))}</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lvl = min(len(m.group(1)) + 1, 6)
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f'<h{lvl} id="{slug}">{md_inline(m.group(2))}</h{lvl}>')
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([md_inline(c.strip()) for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append(f'<div class="table-scroll">{table(hdr, rows)}</div>')
            i = j
            continue
        if re.match(r"^\s*[-*]\s+", ln) or re.match(r"^\s*\d+\.\s+", ln):
            ordered = bool(re.match(r"^\s*\d+\.\s+", ln))
            items, j = [], i
            while j < len(lines) and (re.match(r"^\s*[-*]\s+", lines[j]) or re.match(r"^\s*\d+\.\s+", lines[j])):
                items.append(md_inline(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[j])))
                j += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{x}</li>" for x in items) + f"</{tag}>")
            i = j
            continue
        if ln.startswith(">"):
            buf, j = [], i
            while j < len(lines) and lines[j].startswith(">"):
                buf.append(lines[j].lstrip("> "))
                j += 1
            out.append(f"<blockquote>{md_inline(' '.join(buf))}</blockquote>")
            i = j
            continue
        if ln.strip() in ("---", "***"):
            out.append('<hr class="mast-rule" aria-hidden="true">')
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        buf, j = [], i
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#{1,6}\s|```|\||>|\s*[-*]\s|\s*\d+\.\s)", lines[j]):
            buf.append(lines[j].strip())
            j += 1
        out.append(f"<p>{md_inline(' '.join(buf))}</p>")
        i = j
    return "\n".join(out)


def paper(site: Site):
    p = ROOT / "paper" / "paper.md"
    md = p.read_text() if p.exists() else "# Manuscript\n\nThe manuscript has not been generated yet. Run `make paper`."
    html_body = markdown(md)
    site.page("/paper/", "Paper", "GenderNews Atlas manuscript.", f'<article class="prose paper">{html_body}</article>')


# ----------------------------------------------------------------------------- main
def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    shutil.copytree(ASSETS, DIST / "assets")
    shutil.copytree(DATA, DIST / "data")
    figs = ROOT / "figures"
    if figs.exists():
        shutil.copytree(figs, DIST / "figures")
    (DIST / "annotate").mkdir()
    shutil.copy(ROOT / "tools" / "annotate" / "index.html", DIST / "annotate" / "index.html")
    site = Site()
    for fn in (home, timeline, roles, quotes, language, topics, methods, disagreement, validation, robustness, failures,
               data_page, paper):
        fn(site)
    (DIST / "404.html").write_text((DIST / "index.html").read_text())
    n = len(list(DIST.rglob("index.html")))
    print(f"built {n} pages into {DIST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
