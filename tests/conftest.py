"""Shared fixtures and constants for petey tests."""
import os
from pathlib import Path

# Keep MODELS deterministic across machines — don't pick up the
# developer's ~/.petey/models.yaml during tests. Must be set before
# petey.registries.models is imported.
os.environ.setdefault("PETEY_DISABLE_USER_MODELS", "1")

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"
