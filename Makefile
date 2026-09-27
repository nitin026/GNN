# SIH 26078 research + data foundation. `make all` reproduces everything (seeds are fixed).
# Windows without GNU make: `python scripts/run_all.py <target>` runs the same recipes.
PY ?= .venv/Scripts/python
export PYTHONIOENCODING=utf-8

.PHONY: all research data-real data-synth validate baseline improvements gnn test verify

all: research data-real data-synth validate baseline improvements gnn test verify

research:
	$(PY) scripts/check_links.py --prune

data-real:
	$(PY) scripts/fetch_real.py boundary era5 rh850 era5_850_daily clim ibtracs dem imd keyed repack
	$(PY) scripts/write_sources.py

data-synth:
	$(PY) -m synth.generate

validate:
	$(PY) -m synth.validate

baseline:
	$(PY) -m pipeline.evaluate

improvements:
	$(PY) -m pipeline.tune_tracker2
	$(PY) -m pipeline.phase1_report

gnn:
	$(PY) -m pipeline.graphs
	$(PY) scripts/train_tracker.py --variant full
	$(PY) scripts/train_tracker.py --variant temporal
	$(PY) scripts/train_tracker.py --variant cross
	$(PY) scripts/train_tracker.py --variant none
	$(PY) -m pipeline.gnn_eval

test:
	$(PY) -m pytest -q

verify:
	$(PY) scripts/verify_deliverables.py
