# GenderNews Atlas -- reproduction targets.
# Tiers: GNA_TIER=smoke (CI, <2 min), standard (paper), extended (larger sample).
# Method E needs GNA_LLM_API_KEY; every other target runs offline after `make data`.

PY      := .venv/bin/python
TIER    ?= standard
export GNA_TIER := $(TIER)

.PHONY: setup data preprocess annotate models validate llm analyze robustness figures paper site test lint reproduce smoke freeze clean-interim

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt
	$(PY) -m spacy info en_core_web_sm >/dev/null 2>&1 || uv pip install --python $(PY) \
	  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

data:                      ## stream + sample AmericanStories (CC-BY-4.0) by year
	$(PY) scripts/fetch_corpus.py --tier $(TIER)

audit:                     ## prefix-sampling representativeness audit (downloads one full year)
	$(PY) scripts/audit_prefix_sampling.py

preprocess:                ## quality/dedup, NER + rules extraction, topics
	$(PY) scripts/corpus_quality.py
	$(PY) scripts/extract_persons.py --tier $(TIER)
	$(PY) scripts/topics.py

annotate:                  ## draw the stratified blind reference sample (labels are added by an annotator)
	$(PY) scripts/make_annotation_sample.py
	@echo "Open tools/annotate/index.html and load data/annotation/sample_blind.jsonl"

llm:                       ## method E (optional; needs GNA_LLM_API_KEY)
	$(PY) scripts/run_llm.py --mode reference
	$(PY) scripts/run_llm.py --mode apply

models validate:           ## methods C/D cross-fitted on reference labels; weighted P/R/F1; application
	$(PY) scripts/role_methods.py

analyze:
	$(PY) scripts/build_frame.py
	$(PY) scripts/analyze_trends.py
	$(PY) scripts/analyze_composition.py
	$(PY) scripts/analyze_methods.py
	$(PY) scripts/analyze_language.py

robustness:
	$(PY) scripts/robustness.py
	$(PY) scripts/claims.py

figures:
	$(PY) scripts/figures.py

freeze:
	$(PY) scripts/freeze.py

paper: freeze
	$(PY) scripts/build_paper.py

site:
	$(PY) scripts/export_site_data.py
	cd site && npm ci && npm run build

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

reproduce: data preprocess annotate models analyze robustness figures paper site test

smoke:                     ## tiny end-to-end check used in CI (no network models, no LLM)
	$(MAKE) data TIER=smoke
	$(PY) scripts/extract_persons.py --tier smoke --workers 2
	$(PY) -m pytest -q
