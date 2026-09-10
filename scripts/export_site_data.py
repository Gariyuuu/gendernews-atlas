#!/usr/bin/env python
"""make site (step 1) -- convert frozen results into compact JSON the static site reads.

The site never embeds a hand-typed number: every figure and sentence with a number is
rendered from these files, which are mechanical projections of results/*.parquet and
results/release.json.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.paths import RESULTS, ROOT  # noqa: E402

OUT = ROOT / "site" / "public" / "data"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 5)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def records(df: pd.DataFrame) -> list[dict]:
    return clean(df.to_dict("records"))


def dump(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(clean(obj), ensure_ascii=False, separators=(",", ":")))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(RESULTS / "release.json", OUT / "release.json")
    for f in ("disagreement_examples.json", "topic_model.json", "sampling_audit.json", "claims.json",
              "failure_examples.json", "annotation_consistency.json"):
        if (RESULTS / f).exists():
            shutil.copy(RESULTS / f, OUT / f)
    shutil.copy(ROOT / "config" / "topic_labels.json", OUT / "topic_labels.json")

    dump("corpus", records(pd.read_parquet(RESULTS / "corpus_summary.parquet")))
    dump("extraction", records(pd.read_parquet(RESULTS / "extraction_summary.parquet")))
    dump("gender_consistency", records(pd.read_parquet(RESULTS / "gender_consistency.parquet")))
    rt = pd.read_parquet(RESULTS / "role_trends.parquet")
    dump("role_trends", records(rt[["estimand", "role", "method", "tier", "population", "year", "n", "k", "share",
                                    "ci_lo", "ci_hi"]]))
    dump("trend_summary", records(pd.read_parquet(RESULTS / "trend_summary.parquet")))
    dump("quotes", records(pd.read_parquet(RESULTS / "quote_trends.parquet")))
    ta = pd.read_parquet(RESULTS / "topic_adjusted.parquet")
    dump("composition", records(ta))
    ma = pd.read_parquet(RESULTS / "method_agreement.parquet")
    dump("methods", records(ma))
    v = pd.read_parquet(RESULTS / "validation.parquet")
    dump("validation", records(v))
    rb = pd.read_parquet(RESULTS / "robustness.parquet")
    dump("robustness_summary", records(rb[rb["kind"] == "summary"]))
    cells = rb[rb["kind"] == "cell"][["estimand", "method", "tier", "population", "dedup", "ocr", "papers", "bins",
                                      "content", "adjust", "slope_pp_dec", "ci_lo", "ci_hi", "n"]]
    dump("robustness_cells", records(cells))
    lang = pd.read_parquet(RESULTS / "language.parquet")
    lo = lang[lang["kind"] == "logodds"]
    keep = []
    for (fam, per), g in lo.groupby(["family", "period"]):
        g = g.sort_values("z")
        keep.append(pd.concat([g.head(20), g.tail(20)]))
    dump("language", {"logodds": records(pd.concat(keep)[["family", "period", "word", "n_F", "n_M", "delta", "z", "q"]]),
                      "agency": records(lang[lang["kind"].isin(["agency_yearly", "agency_trend", "positioned_yearly"])]
                                        .dropna(axis=1, how="all"))})
    print("site data:", sorted(p.name for p in OUT.glob("*.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
