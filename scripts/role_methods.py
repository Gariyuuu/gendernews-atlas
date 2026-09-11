#!/usr/bin/env python
"""make models + make validate -- methods C/D/E, cross-fitted validation, and application.

Inputs
  data/annotation/labels_v1.jsonl      reference labels (see annotation_protocol.md)
  data/annotation/sample_key.csv       strata and sampling weights
  data/annotation/llm_reference.jsonl  method E on the reference sample (optional)
  data/interim/extract/entities_*.parquet

Outputs
  results/validation.parquet           method x target weighted precision / recall / F1
  data/interim/method_labels.parquet   C and D role probabilities on the application subsample
  data/interim/reference_oof.parquet   out-of-fold C/D scores on the reference sample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gna.lexicons import AUTHORITY, ROLES  # noqa: E402
from gna.frame import apply_mixed_title_rule  # noqa: E402
from gna.paths import DATA, INTERIM, RESULTS, safe_write  # noqa: E402
from gna.roles_ml import (BowRoleModel, EmbedRoleModel, bow_text, crossfit, encode_contexts,  # noqa: E402
                          roles_matrix)

ANN = DATA / "annotation"
APPLY_PER_YEAR = 5000
ENT_COLS = ["entity_id", "article_id", "year", "publication", "hclass", "n_ner_mentions", "gender_h", "gender_hp",
            "gender_hpn", "titles", "roles_a", "roles_b", "n_quotes_name", "n_quotes_pron", "ctx", "ctx_hl_start",
            "ctx_hl_end"]
TARGETS = ROLES + ["AUTHORITY"]


def load_entities(ids: set | None = None, cols=ENT_COLS) -> pd.DataFrame:
    parts = []
    for p in sorted((INTERIM / "extract").glob("entities_*.parquet")):
        d = pd.read_parquet(p, columns=cols)
        parts.append(d[d["entity_id"].isin(ids)] if ids is not None else d)
    return pd.concat(parts, ignore_index=True)


def wprf(y_true, y_pred, w) -> dict:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = float((w * (y_true & y_pred)).sum())
    fp = float((w * (~y_true & y_pred)).sum())
    fn = float((w * (y_true & ~y_pred)).sum())
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else np.nan
    return {"precision": p, "recall": r, "f1": f, "n_pos_ref": int(y_true.sum()), "n_pred_pos": int(y_pred.sum()),
            "prev_ref_w": float((w * y_true).sum() / w.sum()), "prev_pred_w": float((w * y_pred).sum() / w.sum())}


def add_authority(Y: np.ndarray) -> np.ndarray:
    idx = [ROLES.index(r) for r in AUTHORITY]
    return np.column_stack([Y, Y[:, idx].max(axis=1)])


def gender_rows(method: str, pred, true, w) -> list[dict]:
    pred, true = np.asarray(pred), np.asarray(true)
    cls, ref_cls = pred != "UNKNOWN", true != "UNKNOWN"
    out = [{"method": method, "target": "gender",
            "precision": float((w * (cls & (pred == true))).sum() / (w * cls).sum()) if cls.any() else np.nan,
            "recall": float((w * (ref_cls & (pred == true))).sum() / (w * ref_cls).sum()) if ref_cls.any() else np.nan,
            "f1": np.nan, "n_pos_ref": int(ref_cls.sum()), "n_pred_pos": int(cls.sum()),
            "prev_ref_w": float((w * ref_cls).sum() / w.sum()), "prev_pred_w": float((w * cls).sum() / w.sum())}]
    for g in ("F", "M"):
        out.append({"method": method, "target": f"gender_{g}", **wprf(true == g, pred == g, w)})
    nos = ~ref_cls
    out.append({"method": method, "target": "gender_assigned_without_text_signal",
                "precision": float((w * (nos & cls)).sum() / (w * nos).sum()) if nos.any() else np.nan,
                "recall": np.nan, "f1": np.nan, "n_pos_ref": int(nos.sum()), "n_pred_pos": int((nos & cls).sum()),
                "prev_ref_w": np.nan, "prev_pred_w": np.nan})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-apply", action="store_true")
    ap.add_argument("--labels", default=str(ANN / "labels_v1.jsonl"))
    a = ap.parse_args()

    lab = pd.DataFrame([json.loads(x) for x in open(a.labels) if x.strip()])
    key = pd.read_csv(ANN / "sample_key.csv")
    ref = key.merge(lab, on="entity_id", how="inner")
    ref = ref.merge(load_entities(set(ref["entity_id"])), on="entity_id", how="left")
    w_all = ref["weight"].to_numpy(float)
    print(f"reference labels: {len(ref)}  is_person=yes: {(ref['is_person'] == 'yes').sum()}")

    rows = []
    isp = (ref["is_person"] == "yes").to_numpy()
    rows.append({"method": "NER+rules", "target": "is_person", "precision": float((w_all * isp).sum() / w_all.sum()),
                 "recall": np.nan, "f1": np.nan, "n_pos_ref": int(isp.sum()), "n_pred_pos": len(ref),
                 "prev_ref_w": np.nan, "prev_pred_w": np.nan})

    P = ref[isp].reset_index(drop=True)
    w = P["weight"].to_numpy(float)
    rows += gender_rows("gender_hp_v2", P["gender_hp"], P["gender_text"], w)   # before the mixed-title rule
    P = apply_mixed_title_rule(P)
    for tier in ("gender_h", "gender_hp", "gender_hpn"):
        rows += gender_rows(tier, P[tier], P["gender_text"], w)

    Yfit = roles_matrix(P["roles"])
    Yref = add_authority(Yfit)
    preds = {"A_lexical": add_authority(roles_matrix(P["roles_a"])),
             "B_dependency": add_authority(roles_matrix(P["roles_b"]))}
    texts = [bow_text(c, int(s), int(e)) for c, s, e in zip(P["ctx"], P["ctx_hl_start"], P["ctx_hl_end"])]
    spans = list(zip(P["ctx_hl_start"].astype(int), P["ctx_hl_end"].astype(int)))
    oof_c = crossfit(BowRoleModel, texts, Yfit, w, k=5, seed=0)
    feats = encode_contexts(list(P["ctx"]), spans)
    oof_d = crossfit(EmbedRoleModel, feats, Yfit, w, k=5, seed=0)
    for name, oof in (("C_bow", oof_c), ("D_embed", oof_d)):
        for th in (0.3, 0.5, 0.7):
            preds[f"{name}@{th}"] = add_authority((oof >= th).astype(int))
    oof_df = pd.DataFrame({"entity_id": P["entity_id"]})
    for j, r in enumerate(ROLES):
        oof_df[f"C_{r}"], oof_df[f"D_{r}"] = oof_c[:, j], oof_d[:, j]
    safe_write(lambda t: oof_df.to_parquet(t, index=False), INTERIM / "reference_oof.parquet")

    llm_path = ANN / "llm_reference.jsonl"
    if llm_path.exists():
        E = {d["entity_id"]: d["parsed"] for d in map(json.loads, open(llm_path)) if d.get("parse_ok")}
        has = P["entity_id"].isin(E).to_numpy()
        Pe = P[has]
        pe = add_authority(roles_matrix([E[i]["roles"] for i in Pe["entity_id"]]))
        for j, t in enumerate(TARGETS):
            rows.append({"method": "E_llm", "target": t, **wprf(Yref[has, j], pe[:, j], w[has])})
        rows += gender_rows("E_llm", [E[i]["gender_text"] for i in Pe["entity_id"]], Pe["gender_text"], w[has])
        rows.append({"method": "E_llm", "target": "quoted",
                     **wprf(Pe["quoted"] == "yes", [E[i]["quoted"] for i in Pe["entity_id"]], w[has])})
        ja = TARGETS.index("AUTHORITY")
        for g in ("F", "M"):
            sel = (Pe["gender_hp"] == g).to_numpy()
            rows.append({"method": "E_llm", "target": f"AUTHORITY|{g}",
                         **wprf(Yref[has][sel, ja], pe[sel, ja], w[has][sel])})
        print(f"method E rows on {has.sum()} reference entities")

    for m, Yp in preds.items():
        for j, t in enumerate(TARGETS):
            rows.append({"method": m, "target": t, **wprf(Yref[:, j], Yp[:, j], w)})
    # differential error: AUTHORITY precision/recall within each signalled gender (primary rule)
    ja = TARGETS.index("AUTHORITY")
    for g in ("F", "M"):
        sel = (P["gender_hp"] == g).to_numpy()
        for m, Yp in preds.items():
            rows.append({"method": m, "target": f"AUTHORITY|{g}", **wprf(Yref[sel, ja], Yp[sel, ja], w[sel])})
    qref = P["quoted"] == "yes"
    rows.append({"method": "quote_rule_name", "target": "quoted", **wprf(qref, P["n_quotes_name"] > 0, w)})
    rows.append({"method": "quote_rule_name+pronoun", "target": "quoted",
                 **wprf(qref, (P["n_quotes_name"] + P["n_quotes_pron"]) > 0, w)})

    val = pd.DataFrame(rows)
    val["weighting"] = "sampling-weighted to population"
    val["reference"] = "v1 AI-annotator reference labels (not human)"
    val["n_reference"] = len(P)
    safe_write(lambda t: val.to_parquet(t, index=False), RESULTS / "validation.parquet")
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        show = val[val["target"].isin(["AUTHORITY", "PUBLIC_OFFICE", "CIVIC", "FAMILY", "gender", "quoted", "is_person"])
                   & ~val["method"].str.contains("@0.3|@0.7")]
        print(show[["method", "target", "precision", "recall", "f1", "n_pos_ref"]].round(3).to_string())
    if a.skip_apply:
        return 0

    allE = load_entities()
    allE = allE[(allE["n_ner_mentions"] >= 1) & (allE["gender_hp"] != "UNKNOWN")]
    rng = np.random.default_rng(20260911)
    app = pd.concat([g.iloc[np.sort(rng.choice(len(g), size=min(APPLY_PER_YEAR, len(g)), replace=False))]
                     for _, g in allE.groupby("year")], ignore_index=True)
    pc = BowRoleModel().fit(texts, Yfit, w).predict_proba(
        [bow_text(c, int(s), int(e)) for c, s, e in zip(app["ctx"], app["ctx_hl_start"], app["ctx_hl_end"])])
    fa = encode_contexts(list(app["ctx"]), list(zip(app["ctx_hl_start"].astype(int), app["ctx_hl_end"].astype(int))))
    pdd = EmbedRoleModel().fit(feats, Yfit, w).predict_proba(fa)
    out = app[["entity_id", "article_id", "year", "publication", "gender_h", "gender_hp", "gender_hpn",
               "roles_a", "roles_b"]].copy()
    for j, r in enumerate(ROLES):
        out[f"C_{r}"], out[f"D_{r}"] = pc[:, j].astype(np.float32), pdd[:, j].astype(np.float32)
    safe_write(lambda t: out.to_parquet(t, index=False), INTERIM / "method_labels.parquet")
    print(f"applied C/D to {len(out)} entities ({APPLY_PER_YEAR}/year max)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
