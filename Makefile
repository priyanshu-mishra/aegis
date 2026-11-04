.PHONY: install test lint demo studies model-card site site-test

PY ?= .venv/bin/python

install:  ## Python package with dev tools, and the site's Node dependencies
	$(PY) -m pip install -e ".[dev]"
	cd site && npm ci

test:  ## Offline Python suite, then the browser ports against the Python golden files
	$(PY) -m pytest -q
	cd site && npm test

lint:  ## What CI runs
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

demo:  ## The whole pipeline offline on a synthetic market (about 3 seconds)
	$(PY) main.py demo

studies:  ## The two extension studies: point models and early warning
	$(PY) scripts/compare_point_models.py
	$(PY) scripts/early_warning_study.py

model-card:  ## Regenerate docs/MODEL_CARD.md from site/public/data
	$(PY) main.py model-card -c configs/default.yaml

site:  ## Build the static site into site/dist
	cd site && npm run build

site-test: test site  ## Everything CI checks before it deploys
