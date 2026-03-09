VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: venv install test clean publish

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -e ".[dev]"

test: install
	$(PYTHON) -m pytest tests/ -v

publish:
	rm -rf dist/
	$(PYTHON) -m build
	. .env && $(PYTHON) -m twine upload dist/* -u __token__ -p $$PYPI_API_KEY

clean:
	rm -rf $(VENV) *.egg-info dist/
