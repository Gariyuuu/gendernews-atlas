#!/usr/bin/env python
"""make preprocess (topics) -- NMF topic model with seed-stability check.

Topic modelling is supporting infrastructure for composition adjustment, not a result.
Fits TF-IDF + NMF on a year-balanced subsample, checks component stability across
seeds (Hungarian matching on topic-term cosine), then assigns every article.

Outputs
  data/interim/article_topics.parquet   article_id, topic, topic_weight, topic_margin
  results/topic_model.json              top terms, stability, per-year topic shares
Component labels live in config/topic_labels.json (assigned by reading top terms).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.decomposition import NMF  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from gna.fetch import iter_articles, target_years  # noqa: E402
from gna.paths import CONFIG, INTERIM, RESULTS, load_config, tier  # noqa: E402
from gna.textnorm import normalize  # noqa: E402

K = 24
SEEDS = [11, 22, 33]
PER_YEAR_FIT = 2500


def main() -> int:
    cfg = load_config("corpus")
    years = target_years(cfg, tier())
    ids, yrs, texts = [], [], []
    for a in iter_articles(years):
        ids.append(a["article_id"])
        yrs.append(a["year"])
        texts.append((a["headline"] + " . " + normalize(a["text"])).lower())
    yrs = np.array(yrs)
    rng = np.random.default_rng(7)
    fit_idx = np.concatenate([rng.choice(np.where(yrs == y)[0], size=min(PER_YEAR_FIT, (yrs == y).sum()), replace=False)
                              for y in sorted(set(yrs))])

    vec = TfidfVectorizer(max_features=25000, min_df=15, max_df=0.35, stop_words="english",
                          token_pattern=r"(?u)\b[a-z]{3,}\b", sublinear_tf=True, dtype=np.float32)
    X_fit = vec.fit_transform([texts[i] for i in fit_idx])
    vocab = np.array(vec.get_feature_names_out())

    models = []
    for s in SEEDS:
        m = NMF(n_components=K, init="nndsvdar", random_state=s, max_iter=400, beta_loss="frobenius")
        m.fit(X_fit)
        models.append(m)
        print(f"seed {s} reconstruction_err={m.reconstruction_err_:.2f}", flush=True)

    ref = models[0].components_
    refn = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    stability = []
    for m in models[1:]:
        c = m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True)
        sim = refn @ c.T
        r, cidx = linear_sum_assignment(-sim)
        stability.append(sim[r, cidx])
    stab = np.vstack(stability)                   # (n_other_seeds, K) matched cosines

    X_all = vec.transform(texts)
    W = models[0].transform(X_all)
    srt = np.sort(W, axis=1)
    topic = W.argmax(axis=1)
    weight = srt[:, -1]
    margin = srt[:, -1] - srt[:, -2]
    # assignment agreement across seeds (after matching) on a 20k article probe
    probe = rng.choice(len(texts), size=min(20000, len(texts)), replace=False)
    agree = []
    for m, st in zip(models[1:], stability):
        c = m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True)
        _, cidx = linear_sum_assignment(-(refn @ c.T))
        inv = np.empty(K, dtype=int)
        inv[cidx] = np.arange(K)
        t2 = inv[m.transform(X_all[probe]).argmax(axis=1)]
        agree.append(float((t2 == topic[probe]).mean()))

    zero = (weight == 0)
    topic = np.where(zero, -1, topic)
    pd.DataFrame({"article_id": ids, "year": yrs, "topic": topic, "topic_weight": weight,
                  "topic_margin": margin}).to_parquet(INTERIM / "article_topics.parquet", index=False)

    top_terms = {int(k): vocab[np.argsort(-ref[k])[:18]].tolist() for k in range(K)}
    shares = pd.crosstab(yrs, topic, normalize="index").round(4)
    labels_path = CONFIG / "topic_labels.json"
    out = {
        "k": K, "seeds": SEEDS, "fit_articles": int(len(fit_idx)), "vocab_size": int(len(vocab)),
        "top_terms": top_terms,
        "stability_matched_cosine_mean": {int(k): float(stab[:, k].mean()) for k in range(K)},
        "stability_matched_cosine_overall": float(stab.mean()),
        "assignment_agreement_across_seeds": agree,
        "share_zero_weight_articles": float(zero.mean()),
        "year_topic_shares": {int(y): {int(t): float(v) for t, v in row.items()} for y, row in shares.iterrows()},
        "labels_file": str(labels_path.relative_to(CONFIG.parent)),
    }
    with open(RESULTS / "topic_model.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for k in range(K):
        print(f"{k:2d} stab={stab[:, k].mean():.2f} share={(topic == k).mean():.3f} :: {' '.join(top_terms[k][:14])}")
    print("seed assignment agreement:", [round(a, 3) for a in agree])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
