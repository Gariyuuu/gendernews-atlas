#!/usr/bin/env python
"""make validate (step 3) -- intra-annotator consistency of the v1 reference labels.

A random 20% of the reference sample (data/annotation/consistency_ids.txt) is re-labelled
blind, in shuffled order, after the full pass.  Cohen's kappa between the two passes is
SELF-consistency of one annotator (the AI build agent); it is not inter-annotator
agreement and says nothing about agreement with human judgement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from gna.lexicons import AUTHORITY, ROLES  # noqa: E402
from gna.paths import DATA, RESULTS, safe_write  # noqa: E402

ANN = DATA / "annotation"


def kappa(a, b):
    a, b = list(a), list(b)
    if len(set(a) | set(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def main() -> int:
    one = pd.DataFrame([json.loads(x) for x in open(ANN / "labels_v1.jsonl") if x.strip()])
    two = pd.DataFrame([json.loads(x) for x in open(ANN / "labels_v1_consistency.jsonl") if x.strip()])
    d = one.merge(two, on="entity_id", suffixes=("_1", "_2"))
    rows = []
    for field in ("is_person", "gender_text", "quoted"):
        rows.append({"field": field, "kappa": kappa(d[f"{field}_1"], d[f"{field}_2"]),
                     "agreement": float((d[f"{field}_1"] == d[f"{field}_2"]).mean()), "n": len(d)})
    r1 = d["roles_1"].map(set)
    r2 = d["roles_2"].map(set)
    for role in ROLES + ["AUTHORITY"]:
        want = AUTHORITY if role == "AUTHORITY" else {role}
        a = r1.map(lambda s, w=want: bool(s & w))
        b = r2.map(lambda s, w=want: bool(s & w))
        rows.append({"field": f"role:{role}", "kappa": kappa(a, b), "agreement": float((a == b).mean()), "n": len(d),
                     "positives_pass1": int(a.sum()), "positives_pass2": int(b.sum())})
    out = {"what": ("intra-annotator self-consistency: the same AI annotator re-labelled a 20% subset blind to "
                    "method outputs, in shuffled order, after the full pass, but in the same working session, "
                    "with first-pass labels still in its context history. Treat it as an UPPER BOUND on "
                    "self-consistency, not as independent agreement"),
           "n_items": len(d), "rows": [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in rows]}
    safe_write(lambda t: Path(t).write_text(json.dumps(out, indent=1)), RESULTS / "annotation_consistency.json")
    for r in rows:
        print(f"{r['field']:22s} kappa={r['kappa']:.3f} agree={r['agreement']:.3f}" if r["kappa"] == r["kappa"]
              else f"{r['field']:22s} kappa=n/a agree={r['agreement']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
