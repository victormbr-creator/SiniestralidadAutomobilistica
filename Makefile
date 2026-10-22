PYTHON ?= python3

.PHONY: install validate prepare panel geospatial quality spatial train evaluate results coverage report test all clean
install:
	$(PYTHON) -m pip install -e '.[dev]'
validate:
	$(PYTHON) scripts/validate_inputs.py
prepare:
	$(PYTHON) scripts/prepare_atus.py
panel:
	$(PYTHON) scripts/build_panel.py
geospatial:
	$(PYTHON) scripts/prepare_cdmx_geospatial.py
quality:
	$(PYTHON) scripts/analyze_quality.py
spatial:
	$(PYTHON) scripts/run_spatial.py
train:
	$(PYTHON) scripts/run_cdmx_analysis.py
evaluate:
	@echo "La evaluación se genera dentro de make train para preservar la apertura única del test."
results:
	@echo "Las tablas y figuras CDMX se generan dentro de make train."
coverage:
	$(PYTHON) scripts/run_coverage.py
report:
	$(PYTHON) scripts/build_report.py
test:
	$(PYTHON) -m pytest -q
all: validate prepare panel geospatial train report
clean:
	$(PYTHON) scripts/clean_outputs.py
