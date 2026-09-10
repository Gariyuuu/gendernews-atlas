#!/usr/bin/env python
"""make preprocess -- parse every sampled article and extract person entities.

Writes one parquet pair per year to data/interim/extract/ (resumable):
  entities_<year>.parquet  one row per person-in-article
  articles_<year>.parquet  one row per article (parse-level counts)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from gna.extract import EXTRACT_VERSION, extract_article  # noqa: E402
from gna.fetch import iter_articles, normalize_title, target_years  # noqa: E402
from gna.paths import INTERIM, load_config, safe_write, tier, wait_for_space  # noqa: E402
from gna.textnorm import normalize  # noqa: E402

OUT = INTERIM / "extract"
OUT.mkdir(parents=True, exist_ok=True)
META = ("article_id", "year", "date", "publication", "lccn", "state", "page_number", "legibility", "n_words")
_NLP = None


def _init():
    global _NLP
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _NLP.max_length = 200_000


def _work(batch: list[dict]):
    ents, arts = [], []
    pairs = ((normalize(a["text"]),
              {**{k: a.get(k) for k in META},
               "publication": normalize_title(a.get("publication_raw") or a.get("publication") or "")})
             for a in batch)
    for doc, meta in _NLP.pipe(pairs, as_tuples=True, batch_size=64):
        rows, art = extract_article(doc, meta)
        ents.extend(rows)
        arts.append(art)
    return ents, arts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=tier())
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4) - 3)))
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="articles per year (debug)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    cfg = load_config("corpus")
    years = a.years or target_years(cfg, a.tier)
    with Pool(a.workers, initializer=_init) as pool:
        for y in years:
            ep, ap_ = OUT / f"entities_{y}.parquet", OUT / f"articles_{y}.parquet"
            if ep.exists() and ap_.exists() and not a.force:
                if pd.read_parquet(ap_, columns=["extract_version"])["extract_version"].iloc[0] == EXTRACT_VERSION:
                    print(f"{y} cached", flush=True)
                    continue
            wait_for_space()
            t0 = time.time()
            arts = list(iter_articles([y]))
            if a.limit:
                arts = arts[: a.limit]
            chunks = [arts[i:i + a.chunk] for i in range(0, len(arts), a.chunk)]
            E, A = [], []
            for ents, arts_out in pool.imap_unordered(_work, chunks):
                E.extend(ents)
                A.extend(arts_out)
            de, da = pd.DataFrame(E), pd.DataFrame(A)
            de["extract_version"] = da["extract_version"] = EXTRACT_VERSION
            safe_write(lambda t: de.sort_values("entity_id").to_parquet(t, index=False), ep)
            safe_write(lambda t: da.sort_values("article_id").to_parquet(t, index=False), ap_)
            print(f"{y} articles={len(da):>6} entities={len(de):>7} {time.time() - t0:6.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
