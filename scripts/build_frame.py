#!/usr/bin/env python
"""make analyze (step 1) -- assemble the entity-level analysis frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from gna.frame import CLEAN_DICT_RATE, FRAME, LANG, cap_flag, is_f, page_band, role_flags  # noqa: E402
from gna.paths import CONFIG, INTERIM, safe_write  # noqa: E402

KEEP = ["entity_id", "article_id", "year", "publication", "page_number", "hclass", "n_mentions", "n_ner_mentions",
        "gender_h", "gender_hp", "gender_hpn", "src_hp", "pron_signal", "gnoun_signal", "roles_a", "roles_b",
        "titles", "n_quotes_name", "n_quotes_pron", "n_direct_name", "quote_chars_name", "rel_pos"]
LANG_COLS = ["entity_id", "agent_verbs", "patient_verbs", "poss_nouns", "adjs"]


def main() -> int:
    files = sorted((INTERIM / "extract").glob("entities_*.parquet"))
    ents = pd.concat([pd.read_parquet(p, columns=KEEP) for p in files], ignore_index=True)
    lang = pd.concat([pd.read_parquet(p, columns=LANG_COLS) for p in files], ignore_index=True)
    arts = pd.concat([pd.read_parquet(p) for p in sorted((INTERIM / "extract").glob("articles_*.parquet"))],
                     ignore_index=True)

    meta = pd.read_parquet(INTERIM / "articles_meta.parquet",
                           columns=["article_id", "year", "dict_rate", "legibility", "n_words", "near_dup",
                                    "exact_dup", "publication"])
    meta["in_cap"] = cap_flag(meta)
    meta = meta.drop(columns=["year"])
    top = pd.read_parquet(INTERIM / "article_topics.parquet", columns=["article_id", "topic", "topic_weight"])
    labels = json.load(open(CONFIG / "topic_labels.json"))
    lab = {int(k): v for k, v in labels["topics"].items()}
    top["topic_label"] = top["topic"].map(lambda t: lab.get(int(t), {}).get("label", "unassigned"))
    top["section"] = top["topic"].map(lambda t: lab.get(int(t), {}).get("section", "OTHER"))
    top["is_news"] = top["topic"].map(lambda t: bool(lab.get(int(t), {}).get("is_news", True)))

    df = ents.merge(meta.drop(columns=["publication"]), on="article_id", how="left") \
             .merge(top, on="article_id", how="left")
    miss = df["dict_rate"].isna().mean(), df["topic"].isna().mean()
    print(f"entities={len(df):,}  missing meta={miss[0]:.4f}  missing topic={miss[1]:.4f}")

    for t in ("h", "hp", "hpn"):
        df[f"f_{t}"] = is_f(df[f"gender_{t}"])
    df = pd.concat([df, role_flags(df["roles_a"], "A"), role_flags(df["roles_b"], "B")], axis=1)
    df["page_band"] = df["page_number"].map(page_band)
    df["ner_ok"] = df["n_ner_mentions"] >= 1
    df["ocr_clean"] = (df["legibility"] == "Legible") & (df["dict_rate"] >= CLEAN_DICT_RATE)
    df["evening_star"] = df["publication"] == "evening star"
    df["quoted_name"] = df["n_quotes_name"] > 0
    df["quoted_any"] = (df["n_quotes_name"] + df["n_quotes_pron"]) > 0
    df["period"] = pd.cut(df["year"], [1899, 1915, 1930, 1945, 1963], labels=["1900-15", "1918-30", "1933-45", "1948-63"])
    df["bin5"] = (df["year"] // 5) * 5
    df["decade"] = (df["year"] // 10) * 10
    df = df.drop(columns=["roles_a", "roles_b"])

    safe_write(lambda t: df.to_parquet(t, index=False), FRAME)
    safe_write(lambda t: lang.to_parquet(t, index=False), LANG)
    safe_write(lambda t: arts.to_parquet(t, index=False), INTERIM / "articles_parse.parquet")
    print("frame columns:", len(df.columns), "| NER-supported:", int(df["ner_ok"].sum()),
          "| gender-signalled (hp):", int(df["f_hp"].notna().sum()))
    print(df.groupby("year")["f_hp"].agg(["mean", "count"]).round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
