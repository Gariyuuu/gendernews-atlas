#!/usr/bin/env python
"""make validate (step 2) -- concrete failure cases from the reference sample, for /failures/.

Each example states factually what the reference label says and what the pipeline said.
Excerpts are public-domain newspaper OCR, trimmed around the target person.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from gna.frame import apply_mixed_title_rule  # noqa: E402
from gna.lexicons import AUTHORITY  # noqa: E402
from gna.paths import DATA, RESULTS, safe_write  # noqa: E402
from role_methods import ENT_COLS, load_entities  # noqa: E402

ANN = DATA / "annotation"
PER_KIND = 4


def trim(ctx: str, s: int, e: int, pad: int = 200):
    a, b = max(0, s - pad), min(len(ctx), e + pad)
    pre, post = ("…" if a > 0 else ""), ("…" if b < len(ctx) else "")
    return pre + ctx[a:b] + post, s - a + len(pre), e - a + len(pre)


def main() -> int:
    lab = pd.DataFrame([json.loads(x) for x in open(ANN / "labels_v1.jsonl") if x.strip()])
    ents = apply_mixed_title_rule(load_entities(set(lab["entity_id"]), cols=ENT_COLS + ["src_hp"]))
    d = lab.merge(ents, on="entity_id", how="inner").sample(frac=1.0, random_state=11)
    d["ref_auth"] = d["roles"].map(lambda r: bool(set(r) & AUTHORITY))
    d["b_auth"] = d["roles_b"].map(lambda r: bool(set(r) & AUTHORITY))
    d["a_po"] = d["roles_a"].map(lambda r: "PUBLIC_OFFICE" in set(r))
    d["ref_po"] = d["roles"].map(lambda r: "PUBLIC_OFFICE" in set(r))
    kinds = [
        ("NER: not a person", d[d["is_person"] == "no"],
         lambda r: "Reference: not a reference to a person. Extracted as a person entity."),
        ("Gender: rule contradicts text", d[(d["is_person"] == "yes") & (d["gender_hp"] != "UNKNOWN") & (d["gender_text"] != "UNKNOWN")
                                             & (d["gender_hp"] != d["gender_text"])],
         lambda r: f"Reference gender: {r['gender_text']}. Pipeline (honorific + pronoun): {r['gender_hp']} via {r['src_hp']}."),
        ("Role: authority missed by rules (B)", d[(d["is_person"] == "yes") & d["ref_auth"] & ~d["b_auth"]],
         lambda r: f"Reference roles: {', '.join(r['roles']) or 'none'}. Method B roles: {', '.join(r['roles_b']) or 'none'}."),
        ("Role: authority assigned by rules (B) to a woman the reference gives none",
         d[(d["is_person"] == "yes") & (d["gender_hp"] == "F") & d["b_auth"] & ~d["ref_auth"]],
         lambda r: f"Reference roles: {', '.join(r['roles']) or 'none'}. Method B roles: {', '.join(r['roles_b']) or 'none'}."),
        ("Role: public office inherited by the lexical window (A)", d[(d["is_person"] == "yes") & d["a_po"] & ~d["ref_po"]],
         lambda r: f"Reference roles: {', '.join(r['roles']) or 'none'}. Method A found PUBLIC_OFFICE nearby; method B: {', '.join(r['roles_b']) or 'none'}."),
        ("Quote: attributed speech missed", d[(d["is_person"] == "yes") & (d["quoted"] == "yes") & (d["n_quotes_name"] == 0)],
         lambda r: "Reference: speech is attributed to this person. The named-speaker rule found none."),
    ]
    out = []
    for kind, sub, note in kinds:
        for _, r in sub.head(PER_KIND).iterrows():
            ex, s, e = trim(r["ctx"], int(r["ctx_hl_start"]), int(r["ctx_hl_end"]))
            out.append({"kind": kind, "entity_id": r["entity_id"], "year": int(r["year"]), "publication": r["publication"],
                        "excerpt": ex, "hl_start": s, "hl_end": e, "note": note(r), "reference_notes": r.get("notes", "")})
        print(f"{kind}: {len(sub)} cases in reference sample, {min(len(sub), PER_KIND)} exported")
    payload = {"note": "Failure cases from the v1 reference sample (AI-annotator labels, not human). Public-domain OCR excerpts.",
               "examples": out}
    safe_write(lambda t: Path(t).write_text(json.dumps(payload, ensure_ascii=False, indent=1)), RESULTS / "failure_examples.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
