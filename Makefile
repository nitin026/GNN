# SIH 26078 research + data foundation. `make all` reproduces everything (seeds are fixed).
# Windows without GNU make: `python scripts/run_all.py <target>` runs the same recipes.
PY ?= .venv/Scripts/python
export PYTHONIOENCODING=utf-8

.PHONY: all research data-real data-synth validate baseline test verify

all: research data-real data-synth validate baseline test verify

research:
	$(PY) scripts/check_links.py --prune

data-real:
	$(PY) scripts/fetch_real.py era5 rh850 clim ibtracs dem imd keyed repack
	$(PY) scripts/write_sources.py

data-synth:
	$(PY) -m synth.generate

validate:
	$(PY) -m synth.validate

baseline:
	$(PY) -m pipeline.evaluate

test:
	$(PY) -m pytest -q

verify:
	$(PY) scripts/verify_deliverables.py
