PYTHON ?= python3

.PHONY: install download clean-data features train evaluate test dashboard
install:
	$(PYTHON) -m pip install -e '.[dev]'
download:
	$(PYTHON) scripts/run_download.py
clean-data:
	$(PYTHON) scripts/run_cleaning.py $(FILES)
features:
	$(PYTHON) scripts/run_features.py
train:
	$(PYTHON) scripts/run_training.py
evaluate:
	$(PYTHON) scripts/run_evaluation.py
test:
	$(PYTHON) -m pytest
dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py
