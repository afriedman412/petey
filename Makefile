VENV = venv
SYSTEM_PYTHON ?= /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Release workflow:
#   1. Work on `dev`, commit, push.
#   2. Open PR dev → main on GitHub, merge it.
#   3. `git checkout main && git pull --ff-only`
#   4. `make release` (or `make minor`, or `make release v=X.Y.Z`)
#   5. `make publish`
#   6. `make sync-dev` to bring dev forward with the version bump.
# `make release` will refuse to run from any branch other than main,
# or if local main has diverged from origin/main.

.PHONY: venv install test clean publish release minor sync-dev _check-release-branch

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

_check-release-branch:
	@branch=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$branch" != "main" ]; then \
		echo "ERROR: releases must be made from 'main' (currently on '$$branch')."; \
		echo "Merge dev → main on GitHub, then 'git checkout main && git pull --ff-only'."; \
		exit 1; \
	fi
	@git fetch origin --quiet
	@if ! git merge-base --is-ancestor origin/main HEAD; then \
		echo "ERROR: local main has diverged from origin/main."; \
		echo "Run 'git pull --ff-only' (or reconcile manually) before releasing."; \
		exit 1; \
	fi
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: working tree is not clean. Commit or stash before releasing."; \
		exit 1; \
	fi

release: _check-release-branch
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

# Bring dev forward with whatever main has (typically the version
# bump from the last release). Run after `make publish` so the next
# round of dev work starts from a clean state.
sync-dev:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: working tree is not clean."; exit 1; \
	fi
	git fetch origin
	git checkout dev
	git pull --ff-only
	git merge --ff-only origin/main || git merge origin/main -m "merge main into dev"
	git push origin dev

clean:
	rm -rf $(VENV) *.egg-info dist/
