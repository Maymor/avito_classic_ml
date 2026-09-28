PYTHON ?= .venv/bin/python
DATA_DIR ?= data

.PHONY: env install notebook-install run validate verify test notebook

env:
	python3.11 -m venv .venv

install:
	$(PYTHON) -m pip install --no-cache-dir -r requirements.txt

notebook-install:
	$(PYTHON) -m pip install --no-cache-dir -r requirements-notebook.txt

run:
	$(PYTHON) run.py --data-dir "$(DATA_DIR)"

validate:
	$(PYTHON) run.py --data-dir "$(DATA_DIR)" --validate

verify:
	$(PYTHON) verify.py --data-dir "$(DATA_DIR)"

test:
	$(PYTHON) -m unittest discover -s tests -v

notebook:
	BOT_DATA_DIR="$(DATA_DIR)" $(PYTHON) tools/execute_notebook.py
