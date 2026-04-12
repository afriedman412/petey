"""Tests for schema building, loading, edge cases, enums, and date normalization."""
import re

import pytest

from petey import build_model, load_schema

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


class TestBuildModel:
    def test_simple_string_fields(self):
        spec = {"fields": {"name": {"type": "string", "description": "A name"}}}
        model = build_model(spec)
        instance = model(name="test")
        assert instance.name == "test"

    def test_number_field(self):
        spec = {"fields": {"amount": {"type": "number", "description": "Dollar amount"}}}
        model = build_model(spec)
        instance = model(amount=123.45)
        assert instance.amount == 123.45

    def test_enum_with_values(self):
        spec = {"fields": {"status": {
            "type": "category",
            "values": ["Open", "Closed"],
            "description": "Status",
        }}}
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "status_enum" in str(schema)

    def test_enum_without_values_falls_back_to_string(self):
        spec = {"fields": {"status": {
            "type": "category", "description": "Status",
        }}}
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "status_enum" not in str(schema)
        assert "infer" in str(schema).lower()

    def test_table_mode(self):
        spec = {
            "mode": "table",
            "fields": {"address": {"type": "string", "description": "Addr"}},
        }
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "items" in schema.get("properties", {}) or "items" in schema.get("required", [])

    def test_record_type_array_backwards_compat(self):
        spec = {
            "record_type": "array",
            "fields": {"address": {"type": "string", "description": "Addr"}},
        }
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "items" in schema.get("properties", {}) or "items" in schema.get("required", [])

    def test_nested_array_field(self):
        spec = {"fields": {"items": {
            "type": "array",
            "description": "Line items",
            "fields": {
                "name": {"type": "string", "description": "Item name"},
                "cost": {"type": "number", "description": "Cost"},
            },
        }}}
        model = build_model(spec)
        instance = model(items=[{"name": "Roof", "cost": 100.0}])
        assert len(instance.items) == 1
        assert instance.items[0].name == "Roof"

    def test_mci_schema_builds(self):
        spec = {
            "name": "MCI Cases",
            "record_type": "array",
            "fields": {
                "county": {"type": "string", "description": "County name"},
                "address": {"type": "string", "description": "Building address"},
                "docket_number": {"type": "string", "description": "Docket number"},
                "case_status": {"type": "string", "description": "Case status"},
                "closing_date": {"type": "date", "description": "Closing date"},
                "close_code": {
                    "type": "category",
                    "values": ["GP", "GR", "VO"],
                    "description": "Close code",
                },
                "monthly_mci_incr_per_room": {"type": "number", "description": "Monthly increase per room"},
                "mci_items": {
                    "type": "array",
                    "description": "MCI line items",
                    "fields": {
                        "item_name": {"type": "string", "description": "Improvement description"},
                        "claim_cost": {"type": "number", "description": "Claimed amount"},
                        "allowed_cost": {"type": "number", "description": "Allowed amount"},
                    },
                },
            },
        }
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "items" in schema.get("required", [])


class TestLoadSchema:
    def test_loads_par_schema(self):
        par_path = SCHEMAS_DIR / "par_decision.yaml"
        if not par_path.exists():
            pytest.skip("par_decision.yaml not found")
        model, spec = load_schema(par_path)
        assert spec["name"] == "PAR Decision"
        assert "petitioner" in spec["fields"]


class TestSchemaEdgeCases:
    def test_date_field_is_string(self):
        spec = {"fields": {"d": {"type": "date", "description": "A date"}}}
        model = build_model(spec)
        instance = model(d="2025-01-01")
        assert instance.d == "2025-01-01"

    def test_all_fields_optional(self):
        spec = {"fields": {
            "a": {"type": "string", "description": "A"},
            "b": {"type": "number", "description": "B"},
        }}
        model = build_model(spec)
        instance = model()
        assert instance.a is None
        assert instance.b is None

    def test_model_name_from_spec(self):
        spec = {
            "name": "My Model",
            "fields": {"x": {"type": "string", "description": "X"}},
        }
        model = build_model(spec)
        assert model.__name__ == "My_Model"

    def test_default_model_name(self):
        spec = {"fields": {"x": {"type": "string", "description": "X"}}}
        model = build_model(spec)
        assert model.__name__ == "ExtractedData"

    def test_model_name_valid_for_openai(self):
        """Model name must match OpenAI's function name pattern: ^[a-zA-Z0-9_-]+$"""
        pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
        cases = [
            {"name": "cg_officers.yaml", "fields": {"x": {"type": "string", "description": ""}}},
            {"name": "my schema", "fields": {"x": {"type": "string", "description": ""}}},
            {"name": "test@v2", "fields": {"x": {"type": "string", "description": ""}}},
            {"name": "simple_name", "fields": {"x": {"type": "string", "description": ""}}},
        ]
        for spec in cases:
            model = build_model(spec)
            assert pattern.match(model.__name__), (
                f"Model name {model.__name__!r} from spec name {spec['name']!r} "
                f"is not a valid OpenAI function name"
            )

    def test_array_model_name_valid_for_openai(self):
        """Array wrapper model name must also be valid."""
        pattern = re.compile(r"^[a-zA-Z0-9_-]+$")
        spec = {
            "name": "cg_officers.yaml",
            "record_type": "array",
            "fields": {"x": {"type": "string", "description": ""}},
        }
        model = build_model(spec)
        assert pattern.match(model.__name__), (
            f"Array model name {model.__name__!r} is not a valid OpenAI function name"
        )

    def test_field_names_with_spaces(self):
        """Field names with spaces should build without error."""
        spec = {
            "name": "cg_officers",
            "record_type": "array",
            "fields": {
                "Signal Number": {"type": "number", "description": ""},
                "Date of Rank": {"type": "date", "description": ""},
                "Status Indicator Category": {"type": "string", "description": ""},
            },
        }
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "items" in schema.get("required", [])

    def test_text_warn_threshold_exists(self):
        from petey.extract import TEXT_WARN_THRESHOLD
        assert TEXT_WARN_THRESHOLD == 50_000


class TestEnumCaseInsensitive:
    def test_exact_case(self):
        spec = {"fields": {"status": {
            "type": "enum", "values": ["Open", "Closed"], "description": "",
        }}}
        model = build_model(spec)
        assert model(status="Open").status.value == "Open"

    def test_lowercase_matches(self):
        spec = {"fields": {"status": {
            "type": "enum", "values": ["Open", "Closed"], "description": "",
        }}}
        model = build_model(spec)
        assert model(status="open").status.value == "Open"
        assert model(status="closed").status.value == "Closed"

    def test_uppercase_matches(self):
        spec = {"fields": {"status": {
            "type": "enum", "values": ["Open", "Closed"], "description": "",
        }}}
        model = build_model(spec)
        assert model(status="OPEN").status.value == "Open"

    def test_multiword_enum(self):
        spec = {"fields": {"status": {
            "type": "enum", "values": ["In Progress", "Not Started"], "description": "",
        }}}
        model = build_model(spec)
        assert model(status="in progress").status.value == "In Progress"
        assert model(status="IN PROGRESS").status.value == "In Progress"

    def test_gender_case_insensitive(self):
        spec = {"fields": {"gender": {
            "type": "enum", "values": ["Male", "Female", "Non-binary"], "description": "",
        }}}
        model = build_model(spec)
        assert model(gender="Non-Binary").gender.value == "Non-binary"
        assert model(gender="MALE").gender.value == "Male"
        assert model(gender="female").gender.value == "Female"

    def test_invalid_value_still_fails(self):
        from pydantic import ValidationError
        spec = {"fields": {"status": {
            "type": "enum", "values": ["Open", "Closed"], "description": "",
        }}}
        model = build_model(spec)
        with pytest.raises(ValidationError):
            model(status="invalid")


class TestNormalizeDates:
    """Tests for schema.normalize_dates()."""

    SPEC = {"fields": {
        "filed_date": {"type": "date", "description": "Date filed"},
        "issue_date": {"type": "date", "description": "Issue date"},
        "name": {"type": "string", "description": "Name"},
    }}

    def test_natural_language_dates(self):
        from petey.schema import normalize_dates
        rec = {"filed_date": "December 8, 1986", "issue_date": "November 12, 1986", "name": "Test"}
        normalize_dates(rec, self.SPEC)
        assert rec["filed_date"] == "1986-12-08"
        assert rec["issue_date"] == "1986-11-12"
        assert rec["name"] == "Test"  # non-date field unchanged

    def test_already_iso(self):
        from petey.schema import normalize_dates
        rec = {"filed_date": "1993-12-30", "issue_date": "2013-08-29", "name": "X"}
        normalize_dates(rec, self.SPEC)
        assert rec["filed_date"] == "1993-12-30"
        assert rec["issue_date"] == "2013-08-29"

    def test_uppercase_month(self):
        from petey.schema import normalize_dates
        rec = {"filed_date": "DEC 30 1993", "issue_date": "MAR 16 1987", "name": "X"}
        normalize_dates(rec, self.SPEC)
        assert rec["filed_date"] == "1993-12-30"
        assert rec["issue_date"] == "1987-03-16"

    def test_none_and_empty(self):
        from petey.schema import normalize_dates
        rec = {"filed_date": None, "issue_date": "", "name": "X"}
        normalize_dates(rec, self.SPEC)
        assert rec["filed_date"] is None
        assert rec["issue_date"] == ""

    def test_unparseable_left_unchanged(self):
        from petey.schema import normalize_dates
        rec = {"filed_date": "not a date", "issue_date": "2013-08-29", "name": "X"}
        normalize_dates(rec, self.SPEC)
        assert rec["filed_date"] == "not a date"
        assert rec["issue_date"] == "2013-08-29"

    def test_no_date_fields_in_spec(self):
        from petey.schema import normalize_dates
        spec = {"fields": {"name": {"type": "string", "description": "Name"}}}
        rec = {"name": "Test"}
        normalize_dates(rec, spec)
        assert rec["name"] == "Test"
