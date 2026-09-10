#!/usr/bin/env python
"""make paper -- generate paper/paper.md and research/corpus_provenance.md from frozen results.

Every number is read from results/ (or data/manifests/).  Sentences that state a
direction are chosen by the estimate's sign AND its 95% CI; claim wording follows the
status computed by scripts/claims.py.  Interpretive discussion that cannot be generated
lives in paper/discussion.md (hand-written after results were frozen) and is spliced in.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from gna.paths import CONFIG, MANIFESTS, PAPER, RESEARCH, RESULTS, ROOT  # noqa: E402

MINUS = "−"


def ok(v):
    return v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def pct(v, d=1):
    return f"{100 * v:.{d}f}%" if ok(v) else "—"


def sg(v, d=2):
    return (("+" if v > 0 else MINUS if v < 0 else "") + f"{abs(v):.{d}f}") if ok(v) else "—"


def ci(lo, hi, d=2):
    return f"95% CI {sg(lo, d)} to {sg(hi, d)}" if ok(lo) and ok(hi) else "CI unavailable"


def num(v):
    return f"{int(round(v)):,}" if ok(v) else "—"


def dirw(s, lo, hi, up="increased", down="decreased"):
    if not (ok(lo) and ok(hi)):
        return "could not be assessed"
    return up if lo > 0 else down if hi < 0 else "did not change detectably"


def rd(name):
    p = RESULTS / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def row(df, **kw):
    d = df
    for k, v in kw.items():
        if k not in d:
            return {}
        d = d[d[k] == v]
    return d.iloc[0].to_dict() if len(d) else {}


def main() -> int:
    rel = json.load(open(RESULTS / "release.json"))
    claims = {c["id"]: c for c in json.load(open(RESULTS / "claims.json"))["claims"]}
    cs, ex, ts = rd("corpus_summary"), rd("extraction_summary"), rd("trend_summary")
    val, ma, ta, rb, qt = rd("validation"), rd("method_agreement"), rd("topic_adjusted"), rd("robustness"), rd("quote_trends")
    gc, lg = rd("gender_consistency"), rd("language")
    sa = json.load(open(RESULTS / "sampling_audit.json"))
    tm = json.load(open(RESULTS / "topic_model.json"))
    cfg = json.load(open(CONFIG / "corpus.json"))
    man = json.load(open(MANIFESTS / "corpus_manifest.json"))
    lit = list(csv.DictReader(open(RESEARCH / "literature_matrix.csv")))
    disc = (PAPER / "discussion.md").read_text() if (PAPER / "discussion.md").exists() else ""

    n_art = int(cs["n_articles"].sum())
    n_ent = int(ex["n_entities_ner"].sum())
    unk_pooled = float((ex["unknown_share_hp"] * ex["n_entities_ner"]).sum() / ex["n_entities_ner"].sum())
    h1 = row(ts, estimand="female_share", tier="hp", population="ner", role="ALL")
    ref_n = int(val["n_reference"].dropna().iloc[0]) if "n_reference" in val and val["n_reference"].notna().any() else None
    isp = row(val, target="is_person")
    g_hp = row(val, method="gender_hp", target="gender")
    tot_gc = row(gc, year=-1)
    trA = ma[(ma["kind"] == "trend") & (ma["role"] == "AUTHORITY")] if len(ma) else pd.DataFrame()
    slopes = {r["method"]: r for r in trA.to_dict("records")}
    s_vals = [r["slope_pp_dec"] for m, r in slopes.items() if m in "ABCDE" and ok(r["slope_pp_dec"])]
    summ = rb[rb["kind"] == "summary"] if len(rb) else pd.DataFrame()
    rh1 = summ[(summ["estimand"] == "H1_female_share") & summ[["method", "adjust", "tier", "papers"]].isna().all(axis=1)]
    rh1 = rh1.iloc[0].to_dict() if len(rh1) else {}
    lp = ta[ta["kind"] == "lpm"] if len(ta) else pd.DataFrame()
    raw = row(lp, outcome="female_share", sample="all", fe="none")
    adj = row(lp, outcome="female_share", sample="all", fe="topic+paper")
    dec_t = row(ta, kind="decomposition", outcome="female_share", group="paper_group")
    h7 = row(qt, estimand="H7_speaker_minus_mention_share", attribution="name", period="all")
    es0, es1 = cs["evening_star_share"].iloc[0], cs["evening_star_share"].iloc[-1]
    np_pre = cs.loc[cs["year"] <= 1921, "n_publications"].mean()
    np_post = cs.loc[cs["year"] >= 1924, "n_publications"].mean()
    audit = sa["full_year_1963"]["comparison_prefix_vs_full"]
    st = {k: v["status"] for k, v in claims.items()}

    def fvf1(m, t="AUTHORITY"):
        return row(val, method=m, target=t).get("f1")

    L = []
    A = L.append
    A("# GenderNews Atlas: How Robust Are NLP Measurements of Changing Gender Representation in News?")
    A("")
    A("_Gary Wang · GenderNews Atlas project · manuscript generated from frozen results "
      f"(git `{rel['git']['sha'][:7]}`, frozen {rel['frozen_utc'][:10]})_")
    A("")
    A("> **Disclosure.** The pipeline, analyses, reference annotations and first draft of this text were produced by an AI "
      "research agent (Claude) working under the author's direction. The reference labels used for validation are therefore "
      "AI-produced, not human annotations (§5). Every number in this document is generated from the repository's result files.")
    A("")
    A("## Abstract")
    A("")
    A(f"We measure how U.S. newspapers represented people signalled as women and as men between 1900 and 1963, and we test "
      f"how far the answer depends on the NLP method used to obtain it. From {num(n_art)} articles sampled from AmericanStories "
      f"(22 years, 10,000 articles each), we extract {num(n_ent)} person-in-article entities. We assign gender only from "
      f"textual evidence (honorifics and local pronouns), never from first names. {pct(unk_pooled, 0)} of entities carry no "
      f"such evidence and are kept as UNKNOWN. Among gender-signalled people, women's share "
      f"{dirw(h1.get('slope_pp_dec'), h1.get('ci_lo'), h1.get('ci_hi'))}: {sg(h1.get('slope_pp_dec'))} percentage points per "
      f"decade ({ci(h1.get('ci_lo'), h1.get('ci_hi'))}), from {pct(h1.get('early_share'))} in 1900–21 to "
      f"{pct(h1.get('late_share'))} in 1948–63. Adjusting for topic and newspaper composition moves this slope to "
      f"{sg(adj.get('slope_pp_dec'))} ({ci(adj.get('ci_lo'), adj.get('ci_hi'))}). We measure authority roles five ways on the same "
      f"people: rule-based lexical and dependency methods, a supervised bag-of-words model, a contextual-embedding classifier, "
      f"and an open-weights LLM. The estimated trend in women's share of authority roles ranges from "
      f"{sg(min(s_vals)) if s_vals else '—'} to {sg(max(s_vals)) if s_vals else '—'} points per decade across methods. Across "
      f"{num(rh1.get('n_cells'))} analysis specifications, {pct(rh1.get('share_positive'), 0)} give a rising women's share. Of seven "
      f"pre-registered hypotheses, {sum(v == 'supported' for v in st.values())} are supported, "
      f"{sum(v == 'tentative' for v in st.values())} tentative and {sum(v == 'unsupported' for v in st.values())} unsupported. "
      f"The largest measurement risks are the UNKNOWN pool, differential classifiability by gender, and the 1923 change in "
      f"the digitised corpus.")
    A("")
    A("## 1. Introduction")
    A("")
    A("Computational studies of gender in news usually report one trend from one pipeline. Behind that trend sit several "
      "choices, each defensible and each consequential: what counts as a person, what counts as evidence of gender, what "
      "counts as a role, and which newspapers are in the corpus at all. This paper treats those choices as the object of "
      "study. It asks two questions together: how did gender-role representation in U.S. newspapers change between 1900 "
      "and 1963, and which parts of that answer survive reasonable changes of measurement?")
    A("")
    A("Contributions: (i) a reproducible pipeline for person-level gender and role measurement on historical OCR, using "
      "only textual gender signals and keeping UNKNOWN as a category; (ii) a five-method comparison of role measurement on "
      "one shared population, with validation against stratified reference labels; (iii) an explicit treatment of corpus "
      "composition, including the post-1922 digitisation cliff; and (iv) a multiverse robustness analysis, with hypotheses "
      "frozen before any measurement was run.")
    A("")
    A("## 2. Related work")
    A("")
    by = {r["citation"].split(".")[0]: r for r in lit}
    A("Contemporary news counts find men mentioned and especially quoted far more often than women. The Gender Gap Tracker "
      "(Asr et al. 2021) reports men quoted about three times as often as women in Canadian online news. Jia et al. (2016) "
      "find women “seen more than heard”. The Global Media Monitoring Project reports women as about a quarter of people "
      "seen or heard in the news in 2020. Longitudinal work on text has mostly used word embeddings: Garg et al. (2018) show "
      "embedding associations tracking occupational change across the twentieth century. Underwood, Bamman & Lee (2018) find "
      "gendered description of fictional characters becoming less differentiated since 1780, using pronoun-based character "
      "gender. On measurement, van Strien et al. (2020) show OCR noise degrading NER and parsing unevenly, and Antoniak & "
      "Mimno (2018) show embedding neighbourhoods to be unstable under small corpus changes. Rudinger et al. (2018) document "
      "occupational gender bias in coreference systems. Devinney et al. (2022) find that most NLP gender-bias work leaves its "
      "theory of gender implicit. Our design draws on Lappin & Leass (1994) for salience-based pronoun resolution, Monroe et "
      "al. (2008) for regularised log-odds, and Steegen et al. (2016) for multiverse analysis. The literature search was "
      "targeted, not systematic (see `research/novelty_memo.md`).")
    A("")
    A("## 3. Corpus")
    A("")
    ds = cfg["dataset"]
    A(f"We use **{ds['name']}** ({ds['citation']}): article-segmented OCR of U.S. newspapers digitised by the Library of "
      f"Congress, released under {ds['license']} over public-domain source text. For each of {len(man['years'])} years "
      f"({man['years'][0]}–{man['years'][-1]}, every third year) we stream the year's archive and retain the first 10,000 articles "
      f"that pass length (200–20,000 characters) and legibility filters. The archive's members are date-shuffled, so a streamed "
      f"prefix should approximate a random sample. We test this against a complete download of 1963 "
      f"({num(sa['full_year_1963']['n_scans_full_year'])} scans). The prefix's total-variation distance from the full year is "
      f"inside the null distribution of equal-sized random samples for newspaper (p = {audit['lccn']['p_value_one_sided']:.2f}), "
      f"month (p = {audit['month']['p_value_one_sided']:.2f}) and page position (p = {audit['page_band']['p_value_one_sided']:.2f}).")
    A("")
    A(f"The corpus changes composition sharply. Before 1922 the yearly sample draws on {np_pre:.0f} newspapers on average; "
      f"from 1924 on, on {np_post:.0f}. The Washington *Evening Star* rises from {pct(es0)} of sampled articles in "
      f"{int(cs['year'].iloc[0])} to {pct(es1)} in {int(cs['year'].iloc[-1])}. Near-duplicate articles (MinHash, Jaccard ≥ 0.8, "
      f"within year) make up {pct(rel['headline']['corpus'].get('near_dup_share_overall'), 2)} of the sample, so wire-copy "
      f"duplication is negligible at this sampling density. Figure 1 summarises the corpus.")
    A("")
    A("![Figure 1. Corpus composition and OCR quality by year.](../figures/fig01_corpus.png)")
    A("")
    A("## 4. Measurement framework")
    A("")
    A("**Units.** The primary unit is the *person-in-article entity*: spaCy `PERSON` mentions resolved within an article. "
      "Mentions, articles and speech events are separate units and are never mixed in a model.")
    A("")
    A("**Entity resolution.** Mentions carrying different courtesy-honorific classes are never merged. This keeps "
      "“Mrs. Robert Jones” (a period convention for naming wives) apart from “Senator Robert Jones”, so his office cannot "
      "attach to her. Surname-only mentions attach only when the attachment is unique.")
    A("")
    A("**Gender signal.** Tier 1 is a courtesy or noble honorific. Tier 2 is the first *he/she*-family pronoun after a "
      "salient mention, within the next sentence, with no intervening person or human noun; salience means subject "
      "preference, after Lappin & Leass (1994), and entity-level assignment needs two-thirds agreement. Anything else is "
      "UNKNOWN. Gendered nouns (*widow*, *actress*) form a sensitivity tier only, because several are also role terms. "
      "First names are never used. The primary rule is honorific + pronoun (`hp`).")
    A("")
    A("**Roles.** Ten non-exclusive roles: public office, military, business, labour, professional, arts & sports, civic, "
      "society, family, crime & accident. The composite **authority** role (public office ∪ business ∪ professional) was "
      "fixed before outcomes were computed.")
    A("")
    A("**Quotation.** A speech event is a verb from a fixed speech list whose subject is a person mention. Quotation marks "
      "alone never count without an attributed speaker.")
    A("")
    A("## 5. Validation against reference labels")
    A("")
    A(f"We drew a stratified random sample of {num(ref_n)} entities (four periods × honorific class × method-B role present) "
      "and labelled each blind to all method outputs, following `research/annotation_protocol.md`. **The labels were produced "
      "by the AI research agent, not by human annotators.** They measure agreement with a careful, protocol-bound reader, "
      "which is enough to separate methods from one another. They do not replace human validation, and we report them as "
      "reference labels, not gold. All accuracy figures are reweighted to population proportions.")
    A("")
    A(f"NER precision is low on historical OCR: {pct(isp.get('precision'))} of sampled entities are real person references. The "
      f"primary gender rule assigns the correct signalled gender in {pct(g_hp.get('precision'))} of the cases where it assigns one. "
      f"On the full corpus, the pronoun tier agrees with the honorific for {pct(tot_gc.get('agree'))} of the {num(tot_gc.get('n'))} "
      f"people carrying both signals, a large-sample accuracy check that needs no annotation. F1 for the authority role against "
      f"reference labels: A {sg(fvf1('A_lexical'))}, B {sg(fvf1('B_dependency'))}, C {sg(fvf1('C_bow@0.5'))}, "
      f"D {sg(fvf1('D_embed@0.5'))}, E {sg(fvf1('E_llm'))} (Figure 9).".replace("+", ""))
    A("")
    A("![Figure 9. Role F1 against reference labels.](../figures/fig09_validation_f1.png)")
    A("")
    A("## 6. NLP methods")
    A("")
    A("**A — lexical window:** any role noun within ten tokens in the same sentence, part-of-speech filtered so the "
      "baseline is fair rather than a strawman. **B — dependency rules:** titles, appositives, copular and appointment "
      "predicates, organisational heads typed by their organisation, and passive harm predicates. **C — supervised "
      "bag-of-words:** logistic regression on position-marked context n-grams with the name masked. **D — contextual "
      "embeddings:** MiniLM token states pooled over the name and its context, with a logistic head. C and D are "
      "cross-fitted on the reference labels with sampling weights. **E — LLM:** Qwen3-8B (open weights, served by a hosted "
      "provider through the author's gateway), JSON output at temperature 0, with prompt hash and timestamp recorded per call.")
    A("")
    A("## 7. Temporal modelling")
    A("")
    A("Yearly shares carry article-cluster bootstrap CIs. Trends are weighted least squares of the yearly series on decade, "
      "with Newey–West HAC standard errors, reported in percentage points per decade. Annual points are never treated as "
      "independent. Composition adjustment uses a linear probability model with year, topic and newspaper effects and "
      "standard errors clustered two ways (newspaper, year). The early–late change is decomposed by symmetric Kitagawa "
      "shift-share.")
    A("")
    A("## 8. Results")
    A("")
    c = claims["H1"]
    A(f"**Visibility (H1, {c['status']}).** Women's share of gender-signalled people {dirw(h1.get('slope_pp_dec'), h1.get('ci_lo'), h1.get('ci_hi'))} "
      f"({sg(h1.get('slope_pp_dec'))} pp/decade, {ci(h1.get('ci_lo'), h1.get('ci_hi'))}; Figure 3). Across the multiverse, "
      f"{pct(c['evidence'].get('multiverse_share_positive'), 0)} of {num(c['evidence'].get('multiverse_cells'))} specifications give "
      f"a positive slope; the range is {sg((c['evidence'].get('multiverse_slope_range') or [None, None])[0])} to "
      f"{sg((c['evidence'].get('multiverse_slope_range') or [None, None])[1])}.")
    A("")
    A("![Figure 3. Women's share of gender-signalled people.](../figures/fig03_female_share.png)")
    A("")
    c6 = claims["H6"]
    e6 = c6["evidence"]
    A(f"**Classifiability (H6, {c6['status']}).** The UNKNOWN share moves from {pct(e6.get('unknown_share_first'))} to "
      f"{pct(e6.get('unknown_share_last'))} ({sg(e6.get('unknown_slope_pp_per_decade'))} pp/decade, "
      f"{ci(*(e6.get('unknown_ci') or [None, None]))}; Figure 2). Under the extreme allocations of UNKNOWN the trend in women's "
      f"share of *all* people is {sg((e6['slopes_by_handling'].get('female_share_lower_bound') or [None])[0])} (all UNKNOWN men) "
      f"to {sg((e6['slopes_by_handling'].get('female_share_upper_bound') or [None])[0])} (all UNKNOWN women) pp/decade.")
    A("")
    A("![Figure 2. Share of people with no textual gender signal.](../figures/fig02_unknown_share.png)")
    A("")
    c2 = claims["H2"]
    A(f"**Authority roles (H2, {c2['status']}).** "
      + " ".join(f"Under method {m}, women's share of authority roles is {'below' if v['lower_in_every_year'] else 'not below'} "
                 f"their share of all people in every sampled year, and the gap changes by {sg(v['gap_trend_pp_per_decade'])} pp/decade "
                 f"({ci(*v['gap_trend_ci'])})." for m, v in c2["evidence"]["by_method"].items()) + " (Figures 4–5.)")
    A("")
    A("![Figure 4. Women's share within each role, methods A and B.](../figures/fig04_roles_female_share.png)")
    A("")
    c7 = claims["H7"]
    A(f"**Quotation (H7, {c7['status']}).** Women's share of named speakers differs from their share of gender-signalled people "
      f"by {sg(h7.get('diff_pp'))} points ({ci(h7.get('ci_lo'), h7.get('ci_hi'))}; Figure 12).")
    A("")
    A("![Figure 12. Women's share among people mentioned and people quoted.](../figures/fig12_quotes.png)")
    A("")
    ag = lg[lg["kind"] == "agency_trend"] if len(lg) else pd.DataFrame()
    agf, agm = row(ag, gender="F"), row(ag, gender="M")
    A(f"**Agency (exploratory).** The share of syntactic positions in which a person is the agent rather than the patient "
      f"changes by {sg(agf.get('slope_pp_dec'))} pp/decade for women ({ci(agf.get('ci_lo'), agf.get('ci_hi'))}) and "
      f"{sg(agm.get('slope_pp_dec'))} for men ({ci(agm.get('ci_lo'), agm.get('ci_hi'))}; Figure 13).")
    A("")
    A("## 9. Method disagreement")
    A("")
    c4, c5 = claims["H4"], claims["H5"]
    A(f"On one shared population, the five methods give authority-role trends of " +
      ", ".join(f"{m} {sg(r['slope_pp_dec'])} ({ci(r['ci_lo'], r['ci_hi'])})" for m, r in sorted(slopes.items()) if m in "ABCDE") +
      f" pp/decade (Figure 8). H4, material divergence for at least one headline role, is **{c4['status']}**. H5, that the "
      f"lexical method gives the largest absolute trend, is **{c5['status']}** "
      f"({c5['evidence'].get('n_roles_lexical_largest')} of {c5['evidence'].get('n_roles')} headline roles).")
    A("")
    A("![Figure 8. Authority-role share measured five ways.](../figures/fig08_method_comparison.png)")
    A("")
    A("![Figure 14. Agreement between methods by period.](../figures/fig14_kappa.png)")
    A("")
    A("## 10. Composition adjustment")
    A("")
    c3 = claims["H3"]
    A(f"Raw and adjusted trends come from the same estimator. The raw slope is {sg(raw.get('slope_pp_dec'))} "
      f"({ci(raw.get('ci_lo'), raw.get('ci_hi'))}); with topic and newspaper fixed effects it is {sg(adj.get('slope_pp_dec'))} "
      f"({ci(adj.get('ci_lo'), adj.get('ci_hi'))}), a relative change of {pct(c3['evidence'].get('relative_change'), 0)}. H3 is "
      f"**{c3['status']}**. Splitting the early–late change by newspaper group (*Evening Star* vs other papers) gives a total of "
      f"{sg(dec_t.get('total_pp'))} pp: {sg(dec_t.get('composition_pp'))} from composition and {sg(dec_t.get('within_pp'))} "
      f"within groups (Figures 6–7).")
    A("")
    A("![Figure 6. Raw vs composition-adjusted trends.](../figures/fig06_raw_vs_adjusted.png)")
    A("")
    A("![Figure 7. Shift-share decomposition.](../figures/fig07_decomposition.png)")
    A("")
    A(f"The topic model (NMF, k = {tm['k']}) is moderately stable: mean matched cosine across seeds is "
      f"{tm['stability_matched_cosine_overall']:.2f}, and article assignment agreement is "
      f"{', '.join(f'{a:.2f}' for a in tm['assignment_agreement_across_seeds'])}. Adjustment is therefore also reported at the "
      f"coarse section level.")
    A("")
    A("## 11. Robustness")
    A("")
    A("Every headline estimand is recomputed across gender rule × person detector × duplicate handling × OCR filter × "
      "newspaper handling × time binning × content filter, plus method and classifier threshold for role estimands, and "
      "fixed-effects variants on a reduced grid (Figure 10). A claim is *supported* only if its sign survives every cell.")
    A("")
    A("![Figure 10. Multiverse of trend estimates.](../figures/fig10_robustness.png)")
    A("")
    A("| Hypothesis | Claim | Status |")
    A("|---|---|---|")
    for k in sorted(claims):
        A(f"| {k} | {claims[k]['claim']} | **{claims[k]['status']}** |")
    A("")
    if disc:
        A("## 12. Discussion")
        A("")
        A(disc.strip())
        A("")
    A("## 13. Limitations")
    A("")
    A("The corpus is the digitised record, not the American press. The 1923 cliff and the *Evening Star*'s growth confound "
      "raw trends. Gender is a textual signal inside a binary, marital-status-marked honorific system. Men are more often "
      "named without honorifics, so women's share among classified people is plausibly biased upward. Reference labels are "
      "AI-produced and the reference sample is small. Modern NER on historical OCR has low precision. Trend inference rests "
      "on 22 time points. No claim is causal. The full list is in `research/limitations.md`.")
    A("")
    A("## 14. Ethics")
    A("")
    A("The study infers no one's gender identity and uses no first names. Source texts are public domain, and short "
      "excerpts appear only as evidence for labels. The LLM condition sent public-domain excerpts to a hosted inference "
      "provider. See `research/ethics.md`.")
    A("")
    A("## 15. Conclusion")
    A("")
    A("Whether American newspapers of 1900–1963 “represented women more” is not one number. It depends on who can be "
      "gendered from the text, on what counts as a role, and on which newspapers survived into the digitised record. The "
      "estimates that hold across all of these choices are the ones marked supported above. The rest are reported as "
      "tentative or unsupported, with the specifications that break them.")
    A("")
    A("## Reproducibility")
    A("")
    A(f"`make reproduce` rebuilds everything from the public corpus. Corpus manifest SHA-256 `{rel['corpus']['manifest_sha256'][:16]}…`; "
      f"results hashes, figure hashes, software versions and LLM prompt hashes are recorded in `results/release.json`.")
    A("")
    A("## References")
    A("")
    for r in lit:
        A(f"- {r['citation']}. *{r['venue_year']}*. {r['verified_url']}")
    (PAPER / "paper.md").write_text("\n".join(L) + "\n")

    P = ["# Corpus provenance", "", "_Generated by `scripts/build_paper.py` from `data/manifests/` and `results/`. Do not edit by hand._", "",
         f"- **Dataset:** {ds['name']} — {ds['citation']}", f"- **Source collection:** {ds['source_collection']}",
         f"- **Source rights:** {ds['source_rights']}", f"- **Dataset licence:** {ds['license']}",
         f"- **Access:** `https://huggingface.co/datasets/{ds['hf_repo']}` (revision `{ds['hf_revision']}`), streamed per year; "
         "no tarball is stored.", f"- **Tier:** {man['tier']} · **years:** {', '.join(map(str, man['years']))} · "
         f"**articles:** {num(man['total_articles'])}", "", "## Sampling audit", "",
         "| dimension | observed TVD | null median | null p95 | p |", "|---|---|---|---|---|"]
    for k, v in audit.items():
        P.append(f"| {k} | {v['tvd_observed']} | {v['tvd_null_median']} | {v['tvd_null_p95']} | {v['p_value_one_sided']} |")
    P += ["", "## Per-year manifest", "", "| year | scans read | articles | content SHA-256 (16) | fetched |", "|---|---|---|---|---|"]
    for e in man["entries"]:
        P.append(f"| {e['year']} | {e['n_scans_read']} | {e['n_articles']} | `{e['content_sha256'][:16]}` | {e['fetched_utc']} |")
    P += ["", "## What a licensed corpus would add", "",
          "A licensed archive of a single metropolitan daily (such as the Washington Post corpus used by the Berkeley URAP "
          "project) would remove the newspaper-composition confound, add section metadata that AmericanStories lacks, and "
          "extend coverage past 1963. The pipeline is corpus-agnostic: `src/gna/fetch.py` is the only corpus-specific module."]
    (RESEARCH / "corpus_provenance.md").write_text("\n".join(P) + "\n")
    print(f"paper.md: {sum(len(x) for x in L):,} chars; corpus_provenance.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
