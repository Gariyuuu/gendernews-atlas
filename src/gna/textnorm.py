"""Light, documented OCR normalisation applied before parsing.

Every rule here is conservative and listed in research/measurement_framework.md.
We deliberately do not spell-correct: correction models trained on modern text would
inject exactly the period-inappropriate regularities this project is trying to measure.
"""
from __future__ import annotations

import re

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_NEWLINES = re.compile(r"\s*\n\s*")
_DOUBLE_DOT_TITLE = re.compile(r"\b(Mr|Mrs|Dr|Rev|Messrs|Mme|Jr|Sr)\.{2,}")
_OCR_QUOTES = re.compile(r"''|,,")
_SPACES = re.compile(r"[ \t]{2,}")
_QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"})


def normalize(text: str) -> str:
    """Join line-break hyphenation, flatten newlines, repair common OCR quote/period artefacts."""
    t = _HYPHEN_BREAK.sub(r"\1\2", text or "")
    t = _NEWLINES.sub(" ", t)
    t = t.translate(_QUOTE_MAP)
    t = _DOUBLE_DOT_TITLE.sub(r"\1.", t)
    t = _OCR_QUOTES.sub('"', t)
    t = _SPACES.sub(" ", t)
    return t.strip()


_WORD = re.compile(r"[A-Za-z]+")
_SUFFIXES = ("'s", "s", "es", "ed", "d", "ing", "ly", "er", "est", "ers")


def dictionary_rate(text: str, lexicon: set[str]) -> tuple[float, int]:
    """Share of alphabetic tokens (len >= 3) found in a period dictionary, with light suffix stripping.

    Returns (rate, n_tokens_considered).  A coarse OCR-quality proxy, not a word error rate.
    """
    toks = [w.lower() for w in _WORD.findall(text) if len(w) >= 3]
    if not toks:
        return float("nan"), 0
    hit = 0
    for w in toks:
        if w in lexicon:
            hit += 1
            continue
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 2:
                stem = w[: -len(suf)]
                if stem in lexicon or stem + "e" in lexicon or (len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in lexicon):
                    hit += 1
                    break
    return hit / len(toks), len(toks)
