VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: venv install run clean

venv:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

install: venv

run: venv
	$(VENV)/bin/uvicorn server.app:app --reload

clean:
	rm -rf $(VENV)
