"""Method E: LLM annotation through an OpenAI-compatible endpoint.

A measurement condition, never ground truth.  Every call records the model id, the
prompt hash, temperature, timestamp, latency, and the raw reply.  The endpoint and key
come from the environment (GNA_LLM_BASE_URL, GNA_LLM_API_KEY, GNA_LLM_MODEL) and are
never written to the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .lexicons import ROLES

TEMPERATURE = 0.0
MAX_TOKENS = 220

SYSTEM = (
    "You are a careful annotator of historical U.S. newspaper text (1900-1963). The text is noisy OCR. "
    "You label ONE target person, marked with ⟦ ⟧, using ONLY evidence in the text. Reply with a single JSON "
    "object and nothing else."
)

TEMPLATE = """Target person is marked ⟦like this⟧.

Text ({year}, {publication}):
\"\"\"{text}\"\"\"

Return JSON with keys:
- "is_person": true if the marked span refers to a human individual, else false.
- "gender_text": "F", "M", or "UNKNOWN" -- the gender SIGNALLED IN THIS TEXT by honorifics (Mrs., Miss, Mr., ...),
  pronouns clearly referring to the target, or gendered nouns about the target. Do NOT use first names. If the text
  gives no such signal, answer "UNKNOWN".
- "roles": list of roles the text attributes TO THE TARGET (not to relatives), from:
  PUBLIC_OFFICE (officials, judges, police, candidates), MILITARY, BUSINESS (owners, executives, managers, merchants),
  LABOR (employees, wage/domestic/farm workers), PROFESSIONAL (medicine, science, education, law practice, clergy,
  journalism, experts), ARTS_SPORTS, CIVIC (officers/members of clubs, societies, churches, unions, parties),
  SOCIAL (hostess, guest, debutante, bride, society events), FAMILY (identified via a kin relation, e.g. "wife of"),
  CRIME_ACCIDENT (victim, suspect, defendant, injured, arrested). Organisational heads take the organisation's type
  (club president -> CIVIC). Empty list if none.
- "quoted": true if speech (direct or reported) is attributed to the target in this text.
- "confidence": number from 0 to 1."""


def prompt_hash(model: str) -> str:
    blob = json.dumps({"system": SYSTEM, "template": TEMPLATE, "model": model, "temperature": TEMPERATURE,
                       "max_tokens": MAX_TOKENS, "roles": ROLES}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_JSON = re.compile(r"\{.*\}", re.S)


def parse_reply(raw: str) -> dict | None:
    m = _JSON.search(raw or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    g = str(d.get("gender_text", "UNKNOWN")).upper()
    roles = [r for r in (d.get("roles") or []) if r in ROLES]
    try:
        conf = float(d.get("confidence", "nan"))
    except (TypeError, ValueError):
        conf = float("nan")
    return {"is_person": bool(d.get("is_person", True)), "gender_text": g if g in ("F", "M") else "UNKNOWN",
            "roles": sorted(set(roles)), "quoted": bool(d.get("quoted", False)), "confidence": conf}


class LLMClient:
    def __init__(self):
        self.base = os.environ.get("GNA_LLM_BASE_URL", "https://api.gariyuuu.com/v1").rstrip("/")
        self.key = os.environ.get("GNA_LLM_API_KEY", "")
        self.model = os.environ.get("GNA_LLM_MODEL", "Yuu no Sekai")
        self.model_note = os.environ.get("GNA_LLM_MODEL_NOTE", "Qwen3-8B (open weights) behind self-hosted gateway")
        if not self.key:
            raise RuntimeError("GNA_LLM_API_KEY not set; method E is optional and is skipped without it")

    def label(self, item: dict, retries: int = 3) -> dict:
        body = {"model": self.model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                "reasoning": {"enabled": False},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": TEMPLATE.format(**item)}]}
        err = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = requests.post(f"{self.base}/chat/completions", json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {self.key}"})
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"]["content"]
                parsed = parse_reply(raw)
                return {"entity_id": item["entity_id"], "model": self.model, "model_note": self.model_note,
                        "prompt_hash": prompt_hash(self.model), "temperature": TEMPERATURE,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "latency_s": round(time.time() - t0, 2), "raw": raw, "parsed": parsed,
                        "parse_ok": parsed is not None}
            except Exception as e:  # network / server errors are recorded, not raised
                err = repr(e)
                time.sleep(2 ** attempt)
        return {"entity_id": item["entity_id"], "model": self.model, "prompt_hash": prompt_hash(self.model),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": err, "parse_ok": False}

    def label_many(self, items: list[dict], workers: int = 6):
        with ThreadPoolExecutor(workers) as ex:
            yield from ex.map(self.label, items)
