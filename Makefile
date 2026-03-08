VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: venv install test clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -e ".[dev]"

test: install
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf $(VENV) *.egg-info
