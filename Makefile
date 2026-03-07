VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: venv install clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

install: venv

clean:
	rm -rf $(VENV)
