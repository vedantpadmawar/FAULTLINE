.PHONY: help test eval run clean install

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install        Install project dependencies"
	@echo "  make test           Run pytest test suite"
	@echo "  make eval           Run evaluation suite and generate scoreboard"
	@echo "  make run COMPANY=X  Run end-to-end FAULTLINE analysis on a company"
	@echo "  make clean          Clean cache and temporary files"

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest tests/

eval:
	$(PYTHON) eval/run_all.py

run:
	$(PYTHON) -m faultline.main $(COMPANY)

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
