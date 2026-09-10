#!/usr/bin/env python
"""make preprocess (part) -- article-level quality metadata and per-year corpus summary.

Outputs
  data/interim/articles_meta.parquet  one row per article: metadata, OCR proxies, dup flags
  results/corpus_summary.parquet      one row per year: volume, paper mix, OCR, dupes, length
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.fetch import iter_articles, normalize_title, target_years  # noqa: E402
from gna.paths import INTERIM, RESULTS, load_config, safe_write, tier  # noqa: E402
from gna.quality import dedupe, group_flags, hhi, junk_rate, load_lexicon  # noqa: E402
from gna.textnorm import dictionary_rate, normalize  # noqa: E402


def main() -> int:
    cfg = load_config("corpus")
    years = target_years(cfg, tier())
    lex = load_lexicon()
    rows = []
    for y in years:
        arts = list(iter_articles([y]))
        recs = [(a["article_id"], normalize(a["text"])) for a in arts]
        groups = dedupe(recs, threshold=0.8)
        order = {a["article_id"]: (a["date"], a["page_number"] or 0, a["article_id"]) for a in arts}
        flags = group_flags(groups, order)
        for a, (_, text) in zip(arts, recs):
            dr, ntok = dictionary_rate(text, lex)
            rows.append({
                "article_id": a["article_id"], "year": y, "date": a["date"],
                "publication": normalize_title(a.get("publication_raw") or a["publication"]),
                "lccn": a["lccn"], "state": a["state"], "page_number": a["page_number"],
                "legibility": a["legibility"], "n_chars": a["n_chars"], "n_words": a["n_words"],
                "has_byline": bool(a["byline"]), "dict_rate": dr, "dict_ntok": ntok, "junk_rate": junk_rate(text),
                **flags[a["article_id"]],
            })
        print(f"{y} done ({len(arts)})", flush=True)
    df = pd.DataFrame(rows)
    safe_write(lambda t: df.to_parquet(t, index=False), INTERIM / "articles_meta.parquet")

    summ = []
    for y, g in df.groupby("year"):
        pubs = Counter(g["publication"])
        top, topn = pubs.most_common(1)[0]
        summ.append({
            "year": int(y), "n_articles": len(g), "n_publications": len(pubs), "n_states": g["state"].nunique(),
            "top_publication": top, "top_publication_share": topn / len(g),
            "evening_star_share": float((g["publication"] == "evening star").mean()),
            "publication_hhi": hhi(pubs.values()),
            "median_words": float(g["n_words"].median()), "mean_words": float(g["n_words"].mean()),
            "dict_rate_median": float(g["dict_rate"].median()),
            "dict_rate_p10": float(g["dict_rate"].quantile(0.10)),
            "junk_rate_mean": float(g["junk_rate"].mean()),
            "share_legible": float((g["legibility"] == "Legible").mean()),
            "share_front_page": float((g["page_number"] == 1).mean()),
            "share_byline": float(g["has_byline"].mean()),
            "exact_dup_share": float(g["exact_dup"].mean()),
            "near_dup_share": float(g["near_dup"].mean()),
            "largest_near_group": int(g["near_size"].max()),
        })
    s = pd.DataFrame(summ)
    safe_write(lambda t: s.to_parquet(t, index=False), RESULTS / "corpus_summary.parquet")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(s[["year", "n_articles", "n_publications", "evening_star_share", "median_words",
                 "dict_rate_median", "share_legible", "near_dup_share", "largest_near_group"]].round(3).to_string())
    print("overall near-dup share:", round(float(np.mean(df["near_dup"])), 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
