VENV = venv
SYSTEM_PYTHON ?= /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: venv install test clean publish release minor

venv:
	arch -arm64 $(SYSTEM_PYTHON) -m venv $(VENV)

install: venv
	arch -arm64 $(PIP) install -e ".[dev]"

test: install
	$(PYTHON) -m pytest tests/ -v

publish:
	rm -rf dist/
	$(PYTHON) -m build
	. ../.env && $(PYTHON) -m twine upload dist/* -u __token__ -p $$PYPI_API_KEY

_current_version = $(shell grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
_major = $(word 1,$(subst ., ,$(_current_version)))
_minor = $(word 2,$(subst ., ,$(_current_version)))
_patch = $(word 3,$(subst ., ,$(_current_version)))

release:
	$(eval VERSION := $(or $(v),$(if $(filter minor,$(MAKECMDGOALS)),$(_major).$(shell echo $$(($(_minor)+1))).0,$(_major).$(_minor).$(shell echo $$(($(_patch)+1))))))
	@echo "Releasing v$(VERSION) (was $(_current_version))..."
	sed -i '' 's/^version = ".*"/version = "$(VERSION)"/' pyproject.toml
	git add pyproject.toml
	git commit -m "bump version to $(VERSION)"
	git tag "v$(VERSION)"
	git push origin HEAD "v$(VERSION)"
	gh release create "v$(VERSION)" --generate-notes --title "v$(VERSION)"

minor: release
	@true

clean:
	rm -rf $(VENV) *.egg-info dist/
