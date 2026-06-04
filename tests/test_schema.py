"""Tests for schema building, loading, edge cases, enums, and date normalization."""
import re

import pytest

from petey import build_model

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


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


class TestBooleanType:
    """BPT Spec v1.0 §5.3.5 — boolean type and `bool` alias."""

    def test_boolean_true(self):
        spec = {"fields": {"is_paid": {"type": "boolean", "description": "Is paid"}}}
        model = build_model(spec)
        assert model(is_paid=True).is_paid is True

    def test_boolean_false(self):
        spec = {"fields": {"is_paid": {"type": "boolean", "description": ""}}}
        model = build_model(spec)
        assert model(is_paid=False).is_paid is False

    def test_boolean_nullable(self):
        spec = {"fields": {"is_paid": {"type": "boolean", "description": ""}}}
        model = build_model(spec)
        assert model().is_paid is None

    def test_boolean_coerces_strings(self):
        spec = {"fields": {"flag": {"type": "boolean", "description": ""}}}
        model = build_model(spec)
        assert model(flag="yes").flag is True
        assert model(flag="no").flag is False
        assert model(flag="true").flag is True
        assert model(flag="false").flag is False

    def test_boolean_coerces_ints(self):
        spec = {"fields": {"flag": {"type": "boolean", "description": ""}}}
        model = build_model(spec)
        assert model(flag=1).flag is True
        assert model(flag=0).flag is False

    def test_bool_is_alias_for_boolean(self):
        """`bool` and `boolean` encode identically per §5.3.5."""
        spec = {"fields": {"flag": {"type": "bool", "description": ""}}}
        model = build_model(spec)
        assert model(flag=True).flag is True
        assert model(flag=False).flag is False
        assert model().flag is None


class TestTypeAliases:
    """BPT Spec v1.0 §5.3 — type-name aliases."""

    def test_cat_is_alias_for_category(self):
        spec = {"fields": {"status": {
            "type": "cat", "values": ["Open", "Closed"], "description": "",
        }}}
        model = build_model(spec)
        assert model(status="Open").status.value == "Open"

    def test_enum_is_alias_for_category(self):
        """Already exercised in TestEnumCaseInsensitive — pin it explicitly."""
        spec = {"fields": {"status": {
            "type": "enum", "values": ["A", "B"], "description": "",
        }}}
        model = build_model(spec)
        assert model(status="A").status.value == "A"


class TestParentComposition:
    """BPT Spec v1.0 §5.5 — field composition via `parent`."""

    def test_parent_reference_basic(self):
        spec = {
            "fields": {
                "doc_title": {"type": "string", "description": "Title"},
                "line_items": {"type": "array", "description": "Rows"},
                "description": {
                    "type": "string", "description": "Item desc",
                    "parent": "line_items",
                },
                "amount": {
                    "type": "number", "description": "Item cost",
                    "parent": "line_items",
                },
            }
        }
        model = build_model(spec)
        inst = model(
            doc_title="Receipt",
            line_items=[{"description": "Bread", "amount": 3.5}],
        )
        assert inst.doc_title == "Receipt"
        assert len(inst.line_items) == 1
        assert inst.line_items[0].description == "Bread"
        assert inst.line_items[0].amount == 3.5

    def test_parent_form_equivalent_to_inline(self):
        """Both composition modes produce the same output schema (§5.5)."""
        inline = {"fields": {
            "items": {
                "type": "array",
                "fields": {
                    "name": {"type": "string", "description": ""},
                    "cost": {"type": "number", "description": ""},
                },
            },
        }}
        parented = {"fields": {
            "items": {"type": "array"},
            "name": {"type": "string", "description": "", "parent": "items"},
            "cost": {"type": "number", "description": "", "parent": "items"},
        }}
        inline_schema = build_model(inline).model_json_schema()
        parented_schema = build_model(parented).model_json_schema()
        # Same shape: items array of {name, cost}
        assert inline_schema["properties"]["items"]["anyOf"][0]["type"] == \
               parented_schema["properties"]["items"]["anyOf"][0]["type"]

    def test_parent_undefined_raises(self):
        spec = {"fields": {
            "x": {"type": "string", "parent": "does_not_exist"},
        }}
        with pytest.raises(ValueError, match="not defined at the top level"):
            build_model(spec)

    def test_parent_not_array_raises(self):
        spec = {"fields": {
            "p": {"type": "string"},
            "x": {"type": "string", "parent": "p"},
        }}
        with pytest.raises(ValueError, match="not 'array'"):
            build_model(spec)

    def test_parent_mixed_with_inline_raises(self):
        spec = {"fields": {
            "items": {
                "type": "array",
                "fields": {"a": {"type": "string"}},
            },
            "b": {"type": "string", "parent": "items"},
        }}
        with pytest.raises(ValueError, match="exactly one composition mode"):
            build_model(spec)

    def test_parent_does_not_pollute_top_level(self):
        """Fields with `parent` should NOT appear at the top of the output."""
        spec = {"fields": {
            "items": {"type": "array"},
            "child_a": {"type": "string", "parent": "items"},
        }}
        model = build_model(spec)
        # Top-level fields should be only `items` — child_a moved into items
        assert "child_a" not in model.model_fields
        assert "items" in model.model_fields


class TestRequiredFlag:
    """BPT Spec v1.0 §5.4 — `required` is metadata; output stays nullable."""

    def test_required_field_still_nullable_in_output(self):
        """Per §5.4, `required: true` does NOT change output nullability."""
        spec = {"fields": {
            "name": {"type": "string", "required": True, "description": ""},
        }}
        model = build_model(spec)
        # Must still accept null in output — required is a contract assertion,
        # not a type constraint
        assert model().name is None
        assert model(name="x").name == "x"

    def test_required_preserved_in_spec_dict(self):
        """Downstream consumers can read the flag from the original spec."""
        spec = {"fields": {
            "name": {"type": "string", "required": True, "description": ""},
            "nickname": {"type": "string", "description": ""},
        }}
        build_model(spec)  # should not mutate
        assert spec["fields"]["name"]["required"] is True
        assert "required" not in spec["fields"]["nickname"]


class TestPattern:
    """BPT Spec v1.0 §5.6 — `pattern` shapes output via the LLM prompt.

    Pattern is forwarded to the LLM through the field description so it
    knows the canonical output form. Petey does NOT validate extracted
    values against the pattern at this layer — that belongs to BPT
    runtime validation, which is not implemented yet.
    """

    def test_matching_value_passes_through(self):
        spec = {"fields": {"npi": {
            "type": "string", "description": "NPI",
            "pattern": r"^\d{10}$",
        }}}
        model = build_model(spec)
        assert model(npi="1234567890").npi == "1234567890"

    def test_null_passes_unchanged(self):
        spec = {"fields": {"npi": {
            "type": "string", "description": "",
            "pattern": r"^\d{10}$",
        }}}
        model = build_model(spec)
        assert model().npi is None
        assert model(npi=None).npi is None

    def test_mismatch_passes_through_unchanged(self):
        """No input-gating: non-matching values are NOT coerced to None."""
        spec = {"fields": {"npi": {
            "type": "string", "description": "",
            "pattern": r"^\d{10}$",
        }}}
        model = build_model(spec)
        # The LLM sees the pattern in the description and should produce a
        # matching canonical form. Any non-matching value that arrives here
        # is left intact — validation isn't this layer's job.
        assert model(npi="abc").npi == "abc"

    def test_pattern_in_field_description(self):
        """§5.6.1 — pattern surfaced in description so the LLM sees it."""
        spec = {"fields": {"npi": {
            "type": "string", "description": "Box 33a NPI.",
            "pattern": r"^\d{10}$",
        }}}
        model = build_model(spec)
        schema = model.model_json_schema()
        npi_props = schema["properties"]["npi"]
        # description should mention the pattern
        assert "Box 33a NPI." in npi_props.get("description", "")
        assert r"^\d{10}$" in npi_props.get("description", "")

    def test_pattern_in_description_without_user_desc(self):
        """Pattern is surfaced even when no user-provided description."""
        spec = {"fields": {"npi": {
            "type": "string", "pattern": r"^\d{10}$",
        }}}
        model = build_model(spec)
        schema = model.model_json_schema()
        desc = schema["properties"]["npi"].get("description", "")
        assert r"^\d{10}$" in desc

    def test_build_model_permissive_on_non_string_pattern(self):
        """build_model alone does NOT enforce pattern type rules.

        Spec-conformance (pattern only on strings) lives in
        validate_blueprint, called by load_blueprint. build_model is
        permissive and silently ignores patterns on non-string fields.
        """
        for bad_type in ("number", "boolean", "date", "category", "array"):
            cfg = {"type": bad_type, "pattern": r"^\d+$"}
            if bad_type == "category":
                cfg["values"] = ["a", "b"]
            model = build_model({"fields": {"x": cfg}})  # no raise
            desc = (
                model.model_json_schema()["properties"]["x"]
                .get("description", "")
            )
            assert "pattern" not in desc.lower(), (
                f"Pattern leaked into description for type={bad_type!r}"
            )

    def test_build_model_permissive_on_invalid_regex(self):
        """build_model does NOT compile-check the regex."""
        spec = {"fields": {"x": {
            "type": "string", "pattern": "[unclosed",
        }}}
        model = build_model(spec)  # no raise
        desc = (
            model.model_json_schema()["properties"]["x"]
            .get("description", "")
        )
        # The unchecked pattern is still surfaced in the description
        assert "[unclosed" in desc

    def test_pattern_in_array_child_description(self):
        """Pattern propagates into array row JSON schema, no value gating."""
        spec = {"fields": {"items": {
            "type": "array",
            "fields": {
                "code": {
                    "type": "string", "description": "",
                    "pattern": r"^[A-Z]{3}$",
                },
            },
        }}}
        model = build_model(spec)
        # Non-matching value passes through (no gating)
        assert model(items=[{"code": "abc"}]).items[0].code == "abc"
        # And the row-level field carries the pattern in its description
        schema = model.model_json_schema()
        defs = schema.get("$defs") or schema.get("definitions") or {}
        item_def = next(
            (v for v in defs.values() if "code" in v.get("properties", {})),
            None,
        )
        assert item_def is not None
        code_desc = item_def["properties"]["code"].get("description", "")
        assert r"^[A-Z]{3}$" in code_desc

    def test_pattern_with_parent_composition(self):
        """Pattern survives the parent-reference flattening; no gating."""
        spec = {"fields": {
            "items": {"type": "array"},
            "code": {
                "type": "string", "parent": "items",
                "pattern": r"^[A-Z]{3}$",
            },
        }}
        model = build_model(spec)
        assert model(items=[{"code": "lower"}]).items[0].code == "lower"

    def test_pattern_with_required_flag(self):
        """`pattern` + `required: true` coexist as orthogonal metadata."""
        spec = {"fields": {"npi": {
            "type": "string", "required": True,
            "pattern": r"^\d{10}$",
        }}}
        model = build_model(spec)
        # Required is metadata; null still allowed in model
        assert model().npi is None
        # Pattern doesn't gate input — non-matching passes through
        assert model(npi="bad").npi == "bad"
        assert model(npi="9876543210").npi == "9876543210"
        # spec dict still carries `required` for downstream
        assert spec["fields"]["npi"]["required"] is True


class TestValidateBlueprint:
    """BPT Spec v1.0 — load-time blueprint validation.

    validate_blueprint() is called by load_blueprint() and raises on any
    spec-conformance violation. build_model itself stays permissive — the
    enforcement boundary is the blueprint-loading step.
    """

    def test_valid_blueprint_passes(self):
        from petey.schema import validate_blueprint
        validate_blueprint({"fields": {"npi": {
            "type": "string", "pattern": r"^\d{10}$",
        }}})  # no raise

    def test_pattern_on_non_string_raises(self):
        from petey.schema import validate_blueprint
        for bad_type in ("number", "boolean", "date", "category", "array"):
            spec = {"fields": {"x": {
                "type": bad_type, "pattern": r"^\d+$",
            }}}
            with pytest.raises(ValueError, match="not 'string'"):
                validate_blueprint(spec)

    def test_pattern_on_type_aliases_normalized(self):
        from petey.schema import validate_blueprint
        # `cat` → category → raise
        with pytest.raises(ValueError, match="not 'string'"):
            validate_blueprint({"fields": {"x": {
                "type": "cat", "values": ["A", "B"], "pattern": r"^[AB]$",
            }}})
        # `bool` → boolean → raise
        with pytest.raises(ValueError, match="not 'string'"):
            validate_blueprint({"fields": {"x": {
                "type": "bool", "pattern": r"^.+$",
            }}})

    def test_invalid_regex_raises(self):
        from petey.schema import validate_blueprint
        with pytest.raises(ValueError, match="invalid regex"):
            validate_blueprint({"fields": {"x": {
                "type": "string", "pattern": "[unclosed",
            }}})

    def test_pattern_recurses_into_array_children(self):
        from petey.schema import validate_blueprint
        spec = {"fields": {"items": {
            "type": "array",
            "fields": {
                "qty": {"type": "number", "pattern": r"^\d+$"},
            },
        }}}
        with pytest.raises(ValueError, match="not 'string'"):
            validate_blueprint(spec)

    def test_load_blueprint_invokes_validate(self, tmp_path):
        """A .bpt with a non-string pattern fails at load_blueprint()."""
        from petey.schema import load_blueprint
        bpt = tmp_path / "bad.bpt"
        bpt.write_text(
            "fields:\n"
            "  qty:\n"
            "    type: number\n"
            "    pattern: '^\\d+$'\n"
        )
        with pytest.raises(ValueError, match="not 'string'"):
            load_blueprint(bpt)

    def test_load_blueprint_passes_valid(self, tmp_path):
        from petey.schema import load_blueprint
        bpt = tmp_path / "good.bpt"
        bpt.write_text(
            "fields:\n"
            "  npi:\n"
            "    type: string\n"
            "    description: NPI\n"
            "    pattern: '^\\d{10}$'\n"
        )
        model, spec = load_blueprint(bpt)
        assert "npi" in model.model_fields
        assert spec["fields"]["npi"]["pattern"] == r"^\d{10}$"


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
