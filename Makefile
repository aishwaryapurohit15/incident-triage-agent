PY ?= python3
export PYTHONPATH := src

.PHONY: all data pipeline figures test serve clean

all: data pipeline figures test

data:
	$(PY) -m triage.generate

pipeline:
	$(PY) -m triage.pipeline

figures:
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest -q

serve:
	cd docs && $(PY) -m http.server 8000

clean:
	rm -rf data/processed/* reports/* models/*
