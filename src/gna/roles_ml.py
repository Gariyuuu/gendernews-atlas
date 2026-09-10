"""Role methods C (supervised bag-of-words) and D (contextual-embedding classifier).

Both are one-vs-rest binary classifiers over the 10-role taxonomy, trained on the
reference labels with sampling weights (so predicted probabilities target population
prevalence, not the stratified sample's), and evaluated by K-fold cross-fitting:
every reference entity's score comes from a model that never saw it.

The target person's name is replaced by a placeholder in C so the model cannot learn
names (and hence cannot shortcut through name-gender associations).  D sees the name in
context -- that is what a contextual encoder is for -- and pools token states over the
target span.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .lexicons import ROLES

_TOK = re.compile(r"[A-Za-z]+")
MIN_POS = 4                      # fewer positives than this -> constant prevalence predictor
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def bow_text(ctx: str, hs: int, he: int, n: int = 20) -> str:
    left = _TOK.findall(ctx[:hs].lower())[-n:]
    right = _TOK.findall(ctx[he:].lower())[:n]
    # position-marked copies of the 3 nearest words give the model adjacency cues
    near = [f"l_{w}" for w in left[-3:]] + [f"r_{w}" for w in right[:3]]
    return " ".join(left + ["tgtperson"] + right + near)


class _Const:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = X.shape[0]
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _fit_heads(X, Y, w, C: float, solver: str):
    heads = []
    for j in range(Y.shape[1]):
        y = Y[:, j]
        if y.sum() < MIN_POS or (len(y) - y.sum()) < MIN_POS:
            heads.append(_Const(float(np.average(y, weights=w))))
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver=solver)
        clf.fit(X, y, sample_weight=w)
        heads.append(clf)
    return heads


class BowRoleModel:
    name = "C_bow"

    def fit(self, texts, Y, w):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.heads = _fit_heads(X, Y, w, C=4.0, solver="liblinear")
        return self

    def predict_proba(self, texts):
        X = self.vec.transform(texts)
        return np.column_stack([h.predict_proba(X)[:, 1] for h in self.heads])


class EmbedRoleModel:
    """Logistic heads on [target-span mean ; context mean] MiniLM token states."""
    name = "D_embed"

    def fit(self, feats, Y, w):
        self.sc = StandardScaler().fit(feats)
        self.heads = _fit_heads(self.sc.transform(feats), Y, w, C=0.5, solver="lbfgs")
        return self

    def predict_proba(self, feats):
        X = self.sc.transform(feats)
        return np.column_stack([h.predict_proba(X)[:, 1] for h in self.heads])


def encode_contexts(ctxs, spans, batch_size: int = 64, max_length: int = 256, device: str = "cpu"):
    """Return (n, 2*d) array: mean hidden state over target-span tokens, and over all tokens."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ENCODER)
    mdl = AutoModel.from_pretrained(ENCODER).to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(ctxs), batch_size):
            batch = ctxs[i:i + batch_size]
            enc = tok(batch, return_offsets_mapping=True, truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")
            hid = mdl(**{k: v.to(device) for k, v in enc.items()}).last_hidden_state.cpu()
            att = enc["attention_mask"].bool()
            for b, (hs, he) in enumerate(spans[i:i + batch_size]):
                o = offs[b]
                in_span = (o[:, 0] < he) & (o[:, 1] > hs) & (o[:, 1] > o[:, 0]) & att[b]
                all_tok = att[b] & (o[:, 1] > o[:, 0])
                ctx_vec = hid[b][all_tok].mean(0)
                tgt_vec = hid[b][in_span].mean(0) if in_span.any() else ctx_vec
                out.append(torch.cat([tgt_vec, ctx_vec]).numpy())
    return np.vstack(out).astype(np.float32)


def crossfit(make_model, X, Y, w, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold role probabilities for the reference sample."""
    n = len(Y)
    P = np.zeros_like(Y, dtype=float)
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)):
        Xtr = [X[i] for i in tr] if isinstance(X, list) else X[tr]
        Xte = [X[i] for i in te] if isinstance(X, list) else X[te]
        P[te] = make_model().fit(Xtr, Y[tr], w[tr]).predict_proba(Xte)
    return P


def roles_matrix(role_lists) -> np.ndarray:
    idx = {r: j for j, r in enumerate(ROLES)}
    Y = np.zeros((len(role_lists), len(ROLES)), dtype=int)
    for i, rl in enumerate(role_lists):
        for r in rl:
            if r in idx:
                Y[i, idx[r]] = 1
    return Y
