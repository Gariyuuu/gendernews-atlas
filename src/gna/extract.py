"""Person extraction, gender-signal hierarchy, rule-based roles, quotes and agency.

Units produced (never mixed within one statistical model):

* **mention** -- one person reference: a spaCy PERSON span (``src='ner'``) or, in the
  supplementary detector, a courtesy honorific followed by capitalised tokens that
  NER missed (``src='pattern'``).  Titles are stripped into ``titles``.
* **entity** -- a person-in-article: mentions within one article resolved to the
  same individual (rules in ``_cluster``).  Entities are the primary unit.

Entity resolution is conservative by design.  Mentions carrying different courtesy
honorific classes are never merged, so "Mrs. Robert Jones" and "Senator Robert Jones"
(the period convention of naming a wife by her husband's name) stay two people and
the Senator's title cannot leak onto his wife.  Surname-only mentions attach to an
existing entity only when the attachment is unique.

Role methods implemented here:
  A  lexical window  -- any role noun within +/-10 tokens in the same sentence
  B  dependency rules -- titles, appositives, copular/appointment predicates,
     organisational heads disambiguated by the organisation they head, crime/accident
     predicates.  Relations of *other* people ("wife of Senator X") are not inherited.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from . import lexicons as L

EXTRACT_VERSION = "extract-v2"
MAX_NAME_TOKENS = 6
WINDOW_A = 10
_ALPHA = re.compile(r"[A-Za-z]")
_NAME_TOKEN = re.compile(r"^[A-Z][A-Za-z'\-]*\.?$")
_HUMAN_NOUNS = L.GNOUN_F | L.GNOUN_M | set(L._R["FAMILY"].split()) | {"people", "person", "child", "children"}
_KIN_TITLES = {"father": "M", "brother": "M", "sister": "F", "mother": "F"}


def _w(tok) -> str:
    return tok.text.lower().rstrip(".")


def _is_title(tok) -> bool:
    return _w(tok) in L.ALL_TITLES


@dataclass
class Mention:
    s: int                      # name-part token start
    e: int                      # name-part token end (exclusive)
    src: str                    # 'ner' | 'pattern'
    titles: list = field(default_factory=list)
    hclass: str = "N"           # courtesy-honorific class: F / M / N(one or conflicting)
    given: str = ""
    surname: str = ""
    ent: int = -1
    pronoun: str | None = None
    gnoun: str | None = None


# --------------------------------------------------------------------------- mentions
def _make_mention(doc, s0: int, e0: int, src: str) -> Mention | None:
    s, e = s0, e0
    titles = []
    while s < e - 1:                      # strip leading in-span titles, never emptying the name
        t = doc[s]
        if t.text in (".", ","):
            s += 1
            continue
        if _is_title(t):
            titles.append(t)
            s += 1
            continue
        break
    pre, i = [], s0 - 1                   # contiguous preceding title chain ("Rev. Dr.", "Mrs.")
    while i >= 0 and len(pre) < 3:
        t = doc[i]
        if t.text == ".":
            i -= 1
            continue
        if _is_title(t) and t.text[:1].isupper():
            pre.append(t)
            i -= 1
            continue
        break
    titles = pre[::-1] + titles

    name = [t for t in doc[s:e] if t.text not in (".", ",", "'s", "'")]
    alpha = [t for t in name if _ALPHA.search(t.text)]
    if not alpha or (e - s) > MAX_NAME_TOKENS or not any(t.text[:1].isupper() for t in alpha):
        return None
    words = [re.sub(r"[^a-z\-']", "", t.text.lower()).strip("'-") for t in alpha]
    words = [w for w in words if w]
    surname = next((w for w in reversed(words) if len(w) >= 2 and w not in ("jr", "sr", "ii", "iii")), None)
    if not surname:
        return None
    given = words[0] if len(words) >= 2 and words[0] != surname else ""

    hon = [_w(t) for t in titles if _w(t) in L.COURTESY]
    has_f = any(h in L.HONORIFIC_F for h in hon)
    has_m = any(h in L.HONORIFIC_M for h in hon)
    hclass = "F" if has_f and not has_m else ("M" if has_m and not has_f else "N")
    return Mention(s=s, e=e, src=src, titles=titles, hclass=hclass, given=given, surname=surname)


def find_mentions(doc, use_pattern: bool = True) -> list[Mention]:
    out: list[Mention] = []
    covered: set[int] = set()
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        m = _make_mention(doc, ent.start, ent.end, "ner")
        if m:
            out.append(m)
            covered.update(range(ent.start, ent.end))
            covered.update(t.i for t in m.titles)
    if use_pattern:
        n = len(doc)
        for t in doc:
            if t.i in covered or _w(t) not in L.COURTESY or not t.text[:1].isupper():
                continue
            j = t.i + 1
            if j < n and doc[j].text == ".":
                j += 1
            k = j
            while k < n and k - j < 3 and k not in covered and _NAME_TOKEN.match(doc[k].text) and not _is_title(doc[k]):
                k += 1
            if k > j:
                m = _make_mention(doc, t.i, k, "pattern")
                if m and any(len(x.text.strip(".")) >= 2 for x in doc[m.s:m.e]):
                    out.append(m)
                    covered.update(range(t.i, k))
    out.sort(key=lambda m: m.s)
    return out


# --------------------------------------------------------------------------- resolution
def _cluster(ms: list[Mention]) -> tuple[list[dict], int]:
    ents: list[dict] = []
    index: dict[tuple, int] = {}
    for m in ms:
        if not m.given:
            continue
        k = (m.hclass, m.given, m.surname)
        if k not in index:
            index[k] = len(ents)
            ents.append({"hclass": m.hclass, "given": m.given, "surname": m.surname, "members": []})
        m.ent = index[k]
        ents[m.ent]["members"].append(m)
    n_ambiguous = 0
    for m in ms:
        if m.given:
            continue
        same = [i for i, e in enumerate(ents) if e["surname"] == m.surname and e["hclass"] == m.hclass]
        tgt = None
        if m.hclass in ("F", "M"):
            neutral = [i for i, e in enumerate(ents) if e["surname"] == m.surname and e["hclass"] == "N"]
            if len(same) == 1:
                tgt = same[0]
            elif not same and len(neutral) == 1:
                tgt = neutral[0]
                ents[tgt]["hclass"] = m.hclass
            elif same or neutral:
                n_ambiguous += 1
                m.ent = -2
                continue
        else:
            anyc = [i for i, e in enumerate(ents) if e["surname"] == m.surname]
            if len(anyc) == 1:
                tgt = anyc[0]
            elif len(anyc) > 1:
                n_ambiguous += 1
                m.ent = -2
                continue
        if tgt is None:
            tgt = len(ents)
            ents.append({"hclass": m.hclass, "given": "", "surname": m.surname, "members": []})
        m.ent = tgt
        ents[tgt]["members"].append(m)
    return ents, n_ambiguous


_SALIENT_DEPS = {"nsubj", "nsubjpass", "ROOT", "dobj", "attr", "appos", "conj"}
_SUBJ_DEPS = {"nsubj", "nsubjpass"}


def _salient(doc, m: Mention, ms: list[Mention]) -> bool:
    """Salience filter for pronoun antecedents (after Lappin & Leass-style subject preference).

    A mention cannot claim a following pronoun if it sits in a non-argument position
    (prepositional object, possessor, compound -- e.g. "wife of Senator Jones"), or if a
    *different* entity is the subject of the same sentence while this mention is not.
    """
    root = doc[m.s:m.e].root
    if root.dep_ not in _SALIENT_DEPS:
        return False
    if root.dep_ in _SUBJ_DEPS:
        return True
    sent = doc[m.s].sent
    for o in ms:
        if o.ent != m.ent and sent.start <= o.s < sent.end and doc[o.s:o.e].root.dep_ in _SUBJ_DEPS:
            return False
    return True


def _pronoun_after(doc, m: Mention, owner: dict[int, int]) -> str | None:
    """First gendered pronoun after the mention, within this or the next sentence,
    unless another person mention or a human noun intervenes (nearest-antecedent rule).
    Callers apply ``_salient`` first."""
    sent = doc[m.s].sent
    end = sent.end if sent.end >= len(doc) else doc[sent.end].sent.end
    for j in range(m.e, end):
        o = owner.get(j)
        if o is not None and o != m.ent:
            return None
        t = doc[j]
        lw = t.lower_
        if lw in L.PRONOUN_F:
            return "F"
        if lw in L.PRONOUN_M:
            return "M"
        if t.pos_ == "NOUN" and t.lemma_.lower() in _HUMAN_NOUNS:
            return None
    return None


def _gnoun(doc, m: Mention) -> str | None:
    sig = set()
    for t in m.titles:
        g = _KIN_TITLES.get(_w(t))
        if g and t.text[:1].isupper():
            sig.add(g)
    root = doc[m.s:m.e].root
    cands = [c for c in root.children if c.dep_ in ("appos", "compound") and not (m.s <= c.i < m.e)]
    if root.dep_ == "appos" and not (m.s <= root.head.i < m.e):
        cands.append(root.head)
    for c in cands:
        lem = c.lemma_.lower()
        if lem in L.GNOUN_F:
            sig.add("F")
        elif lem in L.GNOUN_M:
            sig.add("M")
    return sig.pop() if len(sig) == 1 else None


def resolve_gender(hclass: str, pronouns: list, gnouns: list) -> dict:
    """Apply the pre-registered hierarchy.  Returns gender under each tier + source."""
    pr = [p for p in pronouns if p]
    pron_signal, pron_src = "UNKNOWN", "none"
    if pr:
        top, n = Counter(pr).most_common(1)[0]
        pron_signal, pron_src = (top, "pronoun") if n / len(pr) >= 2 / 3 else ("UNKNOWN", "conflict")
    gn = {g for g in gnouns if g}
    gn_signal = gn.pop() if len(gn) == 1 else ("UNKNOWN" if not gn else "CONFLICT")

    if hclass in ("F", "M"):
        return {"gender_h": hclass, "gender_hp": hclass, "gender_hpn": hclass, "src_hp": "honorific",
                "pron_signal": pron_signal, "gnoun_signal": gn_signal}
    hpn = gn_signal if gn_signal in ("F", "M") else pron_signal
    return {"gender_h": "UNKNOWN", "gender_hp": pron_signal, "gender_hpn": hpn, "src_hp": pron_src,
            "pron_signal": pron_signal, "gnoun_signal": gn_signal}


# --------------------------------------------------------------------------- roles
def roles_lexical(doc, m: Mention) -> set[str]:
    sent = doc[m.s].sent
    lo, hi = max(sent.start, m.s - WINDOW_A), min(sent.end, m.e + WINDOW_A)
    out = set()
    for j in range(lo, hi):
        if m.s <= j < m.e:
            continue
        t = doc[j]
        if t.pos_ not in ("NOUN", "PROPN"):
            continue
        for key in (_w(t), t.lemma_.lower()):
            r = L.ROLE_TERMS.get(key)
            if r:
                out.add(r)
                break
    return out


def _org_role(h, head_word: str) -> str | None:
    cands = [c for c in h.children if c.dep_ == "compound"]
    for p in h.children:
        if p.dep_ == "prep" and p.lower_ in ("of", "for", "at"):
            for po in p.children:
                if po.dep_ == "pobj":
                    cands.append(po)
                    cands.extend(x for x in po.subtree if x is not po)
    for c in cands:
        w = c.lemma_.lower().strip(".")
        for role, kws in L.ORG_KEYWORDS.items():
            if w in kws:
                return role
    return L.ORG_HEAD_DEFAULT.get(head_word)


def _head_role(h) -> set[str]:
    out = set()
    lem, w = h.lemma_.lower(), _w(h)
    key = lem if (lem in L.ORG_HEADS or lem in L.ROLE_TERMS) else w
    if key in L.ORG_HEADS:
        r = _org_role(h, key)
        if r:
            out.add(r)
    elif key in L.ROLE_TERMS:
        out.add(L.ROLE_TERMS[key])
    for c in h.children:
        if c.dep_ == "compound":
            cl = c.lemma_.lower()
            if cl in L.ROLE_TERMS and cl not in L.ORG_HEADS and L.ROLE_TERMS[cl] != "FAMILY":
                out.add(L.ROLE_TERMS[cl])
    return out


def roles_dependency(doc, m: Mention) -> set[str]:
    roles = set()
    for t in m.titles:
        r = L.TITLE_ROLE.get(_w(t))
        if r:
            roles.add(r)
    root = doc[m.s:m.e].root
    inside = range(m.s, m.e)
    heads = [c for c in root.children if c.dep_ == "appos" and c.i not in inside]
    if root.dep_ == "appos" and root.head.i not in inside:
        heads.append(root.head)
    for c in root.children:
        if c.dep_ == "compound" and c.i < m.s:
            cl = c.lemma_.lower()
            if cl in L.ROLE_TERMS and cl not in L.ORG_HEADS:
                roles.add(L.ROLE_TERMS[cl])
    v = root.head
    vl = v.lemma_.lower()
    if root.dep_ in ("nsubj", "nsubjpass"):
        if vl in ("be", "become", "remain"):
            heads += [c for c in v.children if c.dep_ in ("attr", "acomp")]
        if vl == "serve":
            heads += [po for p in v.children if p.dep_ == "prep" and p.lower_ == "as"
                      for po in p.children if po.dep_ == "pobj"]
        if vl in L.APPOINT_VERBS:
            heads += [c for c in v.children if c.dep_ in ("oprd", "xcomp", "attr", "dobj") and c.i not in inside]
        if root.dep_ == "nsubjpass" and vl in L.CRIME_PASSIVE_VERBS:
            roles.add("CRIME_ACCIDENT")
        if root.dep_ == "nsubj" and vl in L.CRIME_ACTIVE_VERBS:
            roles.add("CRIME_ACCIDENT")
    if root.dep_ == "dobj" and vl in L.CRIME_OBJECT_VERBS:
        roles.add("CRIME_ACCIDENT")
    for h in heads:
        roles |= _head_role(h)
    return roles


# --------------------------------------------------------------------------- quotes
def _quote_len(sent) -> int:
    txt = sent.text
    idx = [i for i, ch in enumerate(txt) if ch == '"']
    if len(idx) >= 2:
        return sum(idx[k + 1] - idx[k] - 1 for k in range(0, len(idx) - 1, 2))
    if len(idx) == 1:
        return max(len(txt) - idx[0] - 1, 0)
    return 0


def quote_events(doc, ms: list[Mention], owner: dict[int, int], ent_gender: dict[int, str]) -> list[dict]:
    """Attributed speech events: a speech verb whose nominal subject is a person mention
    (``how='name'``) or a he/she resolved to the nearest preceding same-gender entity
    within the previous sentence (``how='pronoun'``)."""
    ev = []
    sents = list(doc.sents)
    sent_idx = {s.start: k for k, s in enumerate(sents)}
    for tok in doc:
        if tok.pos_ != "VERB" or tok.lemma_.lower() not in L.SPEECH_VERBS:
            continue
        subj = [c for c in tok.children if c.dep_ == "nsubj"]
        if not subj:
            continue
        s = subj[0]
        eid, how = owner.get(s.i), "name"
        if eid is None and s.lower_ in ("he", "she"):
            want = "F" if s.lower_ == "she" else "M"
            k_here = sent_idx.get(s.sent.start, 0)
            eid, how = None, "pronoun"
            for m in reversed(ms):
                if m.e > s.i or m.ent < 0:
                    continue
                if sent_idx.get(doc[m.s].sent.start, -9) < k_here - 1:
                    break
                if not _salient(doc, m, ms):
                    continue
                if ent_gender.get(m.ent) == want:
                    eid = m.ent
                    break
        if eid is None or eid < 0:
            continue
        sent = tok.sent
        k = sent_idx.get(sent.start, 0)
        direct = '"' in sent.text
        qlen = _quote_len(sent)
        if not direct and len(sent) < 10 and k > 0 and '"' in sents[k - 1].text:
            direct, qlen = True, _quote_len(sents[k - 1])
        if not direct:
            cc = [c for c in tok.children if c.dep_ == "ccomp"]
            qlen = sum(len(doc[c.left_edge.i:c.right_edge.i + 1].text) for c in cc)
        ev.append({"ent": eid, "how": how, "direct": direct, "chars": qlen, "verb": tok.lemma_.lower()})
    return ev


# --------------------------------------------------------------------------- article
def _ctx(doc, text: str, m: Mention) -> tuple[str, int, int]:
    sent = doc[m.s].sent
    a, b = max(0, sent.start_char - 160), min(len(text), sent.end_char + 160)
    while a > 0 and not text[a - 1].isspace():
        a -= 1
    while b < len(text) and not text[b].isspace():
        b += 1
    hs = (m.titles[0].idx if m.titles else doc[m.s].idx)
    he = doc[m.e - 1].idx + len(doc[m.e - 1].text)
    return text[a:b], hs - a, he - a


def extract_article(doc, meta: dict, use_pattern: bool = True) -> tuple[list[dict], dict]:
    text = doc.text
    ms = find_mentions(doc, use_pattern=use_pattern)
    ents, n_amb = _cluster(ms)
    ms = [m for m in ms if m.ent >= 0]
    owner: dict[int, int] = {}
    for m in ms:
        for j in range(m.s, m.e):
            owner[j] = m.ent
        for t in m.titles:
            owner[t.i] = m.ent
    for m in ms:
        m.pronoun = _pronoun_after(doc, m, owner) if _salient(doc, m, ms) else None
        m.gnoun = _gnoun(doc, m)

    genders = {}
    for k, e in enumerate(ents):
        mem = e["members"]
        genders[k] = resolve_gender(e["hclass"], [m.pronoun for m in mem], [m.gnoun for m in mem])
    quotes = quote_events(doc, ms, owner, {k: g["gender_hp"] for k, g in genders.items()})

    rows = []
    for k, e in enumerate(ents):
        mem = sorted(e["members"], key=lambda m: m.s)
        if not mem:
            continue
        ra, rb, titles = set(), set(), []
        agent, patient, poss, adjs = [], [], [], []
        for m in mem:
            ra |= roles_lexical(doc, m)
            rb |= roles_dependency(doc, m)
            titles += [_w(t) for t in m.titles]
            root = doc[m.s:m.e].root
            hl = root.head.lemma_.lower()
            if root.dep_ == "nsubj" and root.head.pos_ in ("VERB", "AUX") and hl != "be":
                agent.append(hl)
            elif root.dep_ in ("nsubjpass", "dobj"):
                patient.append(hl)
            elif root.dep_ == "poss":
                poss.append(hl)
            heads = [c for c in root.children if c.dep_ == "appos"] + [root]
            adjs += [a.lemma_.lower() for h in heads for a in h.children
                     if a.dep_ == "amod" and a.pos_ == "ADJ" and not (m.s <= a.i < m.e)]
        qs = [q for q in quotes if q["ent"] == k]
        first = mem[0]
        ctx, hs, he = _ctx(doc, text, first)
        display = max((doc[m.s:m.e].text for m in mem), key=len)
        g = genders[k]
        rows.append({
            **meta,
            "entity_id": f"{meta['article_id']}#{k}",
            "name": display, "surname": e["surname"], "given": e["given"], "hclass": e["hclass"],
            "n_mentions": len(mem), "n_ner_mentions": sum(m.src == "ner" for m in mem),
            **g,
            "n_pron_f": sum(m.pronoun == "F" for m in mem), "n_pron_m": sum(m.pronoun == "M" for m in mem),
            "roles_a": sorted(ra), "roles_b": sorted(rb), "titles": sorted(set(titles)),
            "n_quotes_name": sum(q["how"] == "name" for q in qs),
            "n_quotes_pron": sum(q["how"] == "pronoun" for q in qs),
            "n_direct_name": sum(q["how"] == "name" and q["direct"] for q in qs),
            "quote_chars_name": sum(q["chars"] for q in qs if q["how"] == "name"),
            "agent_verbs": agent, "patient_verbs": patient, "poss_nouns": poss, "adjs": sorted(set(adjs)),
            "first_tok": first.s, "rel_pos": round(first.s / max(len(doc), 1), 4),
            "ctx": ctx, "ctx_hl_start": hs, "ctx_hl_end": he,
        })
    art = {
        **meta,
        "n_tokens": len(doc), "n_sents": sum(1 for _ in doc.sents),
        "n_ner_person": sum(1 for x in doc.ents if x.label_ == "PERSON"),
        "n_mentions": len(ms), "n_pattern_mentions": sum(m.src == "pattern" for m in ms),
        "n_entities": len(rows), "n_ambiguous_dropped": n_amb, "n_speech_events": len(quotes),
    }
    return rows, art
