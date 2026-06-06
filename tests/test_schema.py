"""Tests for schema building, loading, edge cases, enums, and date normalization."""
import re

import pytest

from petey import build_model

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


def _row(model_cls):
    """Return the row (item) sub-model from an items-wrapped build_model output.

    BPT Spec v1.0 §6 — every build_model output wraps the record(s) in an
    `items` list. These tests check the row-level shape, so we pull the
    inner model class out and operate on it directly.
    """
    return model_cls.model_fields["items"].annotation.__args__[0]


def _row_props(model_cls):
    """Return the JSON schema `properties` dict for the row sub-model."""
    return _row(model_cls).model_json_schema().get("properties", {})


class TestBuildModel:
    def test_simple_string_fields(self):
        spec = {"fields": {"name": {"type": "string", "description": "A name"}}}
        model = build_model(spec)
        instance = model(items=[{"name": "test"}])
        assert instance.items[0].name == "test"

    def test_number_field(self):
        spec = {"fields": {"amount": {"type": "number", "description": "Dollar amount"}}}
        model = build_model(spec)
        instance = model(items=[{"amount": 123.45}])
        assert instance.items[0].amount == 123.45

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
        with pytest.warns(UserWarning, match="no `values:`"):
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
        assert "items" in schema.get("required", [])

    def test_record_type_array_backwards_compat(self):
        spec = {
            "record_type": "array",
            "fields": {"address": {"type": "string", "description": "Addr"}},
        }
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "items" in schema.get("required", [])

    def test_single_record_also_wraps_in_items(self):
        """Unified output (§6): single-record blueprints also use `items`."""
        spec = {"fields": {"name": {"type": "string", "description": ""}}}
        model = build_model(spec)
        schema = model.model_json_schema()
        assert "items" in schema.get("properties", {})
        assert "items" in schema.get("required", [])
        inst = model(items=[{"name": "X"}])
        assert inst.items[0].name == "X"

    def test_nested_array_field(self):
        spec = {"fields": {"line_items": {
            "type": "array",
            "description": "Line items",
            "fields": {
                "name": {"type": "string", "description": "Item name"},
                "cost": {"type": "number", "description": "Cost"},
            },
        }}}
        model = build_model(spec)
        instance = model(items=[{
            "line_items": [{"name": "Roof", "cost": 100.0}],
        }])
        assert len(instance.items[0].line_items) == 1
        assert instance.items[0].line_items[0].name == "Roof"

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
        instance = model(items=[{"d": "2025-01-01"}])
        assert instance.items[0].d == "2025-01-01"

    def test_all_fields_optional(self):
        spec = {"fields": {
            "a": {"type": "string", "description": "A"},
            "b": {"type": "number", "description": "B"},
        }}
        model = build_model(spec)
        instance = model(items=[{}])
        assert instance.items[0].a is None
        assert instance.items[0].b is None

    def test_model_name_from_spec(self):
        spec = {
            "name": "My Model",
            "fields": {"x": {"type": "string", "description": "X"}},
        }
        model = build_model(spec)
        # Wrapper is "<row>List"; row keeps the spec name
        assert model.__name__ == "My_ModelList"
        assert _row(model).__name__ == "My_Model"

    def test_default_model_name(self):
        spec = {"fields": {"x": {"type": "string", "description": "X"}}}
        model = build_model(spec)
        assert model.__name__ == "ExtractedDataList"
        assert _row(model).__name__ == "ExtractedData"

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
        row = _row(build_model(spec))
        assert row(status="Open").status.value == "Open"

    def test_lowercase_matches(self):
        spec = {"fields": {"status": {
            "type": "enum", "values": ["Open", "Closed"], "description": "",
        }}}
        row = _row(build_model(spec))
        assert row(status="open").status.value == "Open"
        assert row(status="closed").status.value == "Closed"

    def test_uppercase_matches(self):
        spec = {"fields": {"status": {
            "type": "enum", "values": ["Open", "Closed"], "description": "",
        }}}
        row = _row(build_model(spec))
        assert row(status="OPEN").status.value == "Open"

    def test_multiword_enum(self):
        spec = {"fields": {"status": {
            "type": "enum", "values": ["In Progress", "Not Started"], "description": "",
        }}}
        row = _row(build_model(spec))
        assert row(status="in progress").status.value == "In Progress"
        assert row(status="IN PROGRESS").status.value == "In Progress"

    def test_gender_case_insensitive(self):
        spec = {"fields": {"gender": {
            "type": "enum", "values": ["Male", "Female", "Non-binary"], "description": "",
        }}}
        row = _row(build_model(spec))
        assert row(gender="Non-Binary").gender.value == "Non-binary"
        assert row(gender="MALE").gender.value == "Male"
        assert row(gender="female").gender.value == "Female"

    def test_invalid_value_still_fails(self):
        from pydantic import ValidationError
        spec = {"fields": {"status": {
            "type": "enum", "values": ["Open", "Closed"], "description": "",
        }}}
        row = _row(build_model(spec))
        with pytest.raises(ValidationError):
            row(status="invalid")


class TestBooleanType:
    """BPT Spec v1.0 §5.3.5 — boolean type and `bool` alias."""

    def test_boolean_true(self):
        spec = {"fields": {"is_paid": {"type": "boolean", "description": "Is paid"}}}
        row = _row(build_model(spec))
        assert row(is_paid=True).is_paid is True

    def test_boolean_false(self):
        spec = {"fields": {"is_paid": {"type": "boolean", "description": ""}}}
        row = _row(build_model(spec))
        assert row(is_paid=False).is_paid is False

    def test_boolean_nullable(self):
        spec = {"fields": {"is_paid": {"type": "boolean", "description": ""}}}
        row = _row(build_model(spec))
        assert row().is_paid is None

    def test_boolean_coerces_strings(self):
        spec = {"fields": {"flag": {"type": "boolean", "description": ""}}}
        row = _row(build_model(spec))
        assert row(flag="yes").flag is True
        assert row(flag="no").flag is False
        assert row(flag="true").flag is True
        assert row(flag="false").flag is False

    def test_boolean_coerces_ints(self):
        spec = {"fields": {"flag": {"type": "boolean", "description": ""}}}
        row = _row(build_model(spec))
        assert row(flag=1).flag is True
        assert row(flag=0).flag is False

    def test_bool_is_alias_for_boolean(self):
        """`bool` and `boolean` encode identically per §5.3.5."""
        spec = {"fields": {"flag": {"type": "bool", "description": ""}}}
        row = _row(build_model(spec))
        assert row(flag=True).flag is True
        assert row(flag=False).flag is False
        assert row().flag is None


class TestTypeAliases:
    """BPT Spec v1.0 §5.3 — type-name aliases."""

    def test_cat_is_alias_for_category(self):
        spec = {"fields": {"status": {
            "type": "cat", "values": ["Open", "Closed"], "description": "",
        }}}
        row = _row(build_model(spec))
        assert row(status="Open").status.value == "Open"

    def test_enum_is_alias_for_category(self):
        """Already exercised in TestEnumCaseInsensitive — pin it explicitly."""
        spec = {"fields": {"status": {
            "type": "enum", "values": ["A", "B"], "description": "",
        }}}
        row = _row(build_model(spec))
        assert row(status="A").status.value == "A"


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
        row = _row(build_model(spec))
        inst = row(
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
            "rows": {
                "type": "array",
                "fields": {
                    "name": {"type": "string", "description": ""},
                    "cost": {"type": "number", "description": ""},
                },
            },
        }}
        parented = {"fields": {
            "rows": {"type": "array"},
            "name": {"type": "string", "description": "", "parent": "rows"},
            "cost": {"type": "number", "description": "", "parent": "rows"},
        }}
        # Same row shape: the rows array's row-item schema should match
        inline_row_schema = _row(build_model(inline)).model_json_schema()
        parented_row_schema = _row(build_model(parented)).model_json_schema()
        # Walk both schemas' rows-item sub-schemas via $defs and compare
        # just the property keys (ignore the generated class names).
        def _rows_item_props(schema):
            defs = schema.get("$defs") or schema.get("definitions") or {}
            for d in defs.values():
                if {"name", "cost"} <= set(d.get("properties", {})):
                    return set(d["properties"])
            return set()

        assert _rows_item_props(inline_row_schema) == {"name", "cost"}
        assert _rows_item_props(parented_row_schema) == {"name", "cost"}

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
            "rows": {
                "type": "array",
                "fields": {"a": {"type": "string"}},
            },
            "b": {"type": "string", "parent": "rows"},
        }}
        with pytest.raises(ValueError, match="exactly one composition mode"):
            build_model(spec)

    def test_parent_does_not_pollute_top_level(self):
        """Fields with `parent` should NOT appear at the row top level."""
        spec = {"fields": {
            "rows": {"type": "array"},
            "child_a": {"type": "string", "parent": "rows"},
        }}
        row = _row(build_model(spec))
        # Row's top-level fields: only `rows` — child_a moved into rows
        assert "child_a" not in row.model_fields
        assert "rows" in row.model_fields


class TestRequiredFlag:
    """BPT Spec v1.0 §5.4 — `required` is metadata; output stays nullable."""

    def test_required_field_still_nullable_in_output(self):
        """Per §5.4, `required: true` does NOT change output nullability."""
        spec = {"fields": {
            "name": {"type": "string", "required": True, "description": ""},
        }}
        row = _row(build_model(spec))
        # Must still accept null in output — required is a contract assertion,
        # not a type constraint
        assert row().name is None
        assert row(name="x").name == "x"

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
    knows the canonical output form. Allowed on `type: string`, `type:
    number`, and `type: category` without `values:` (which compiles to
    string). Petey does NOT validate extracted values against the
    pattern at this layer.
    """

    def test_matching_value_passes_through(self):
        spec = {"fields": {"npi": {
            "type": "string", "description": "NPI",
            "pattern": r"^\d{10}$",
        }}}
        row = _row(build_model(spec))
        assert row(npi="1234567890").npi == "1234567890"

    def test_null_passes_unchanged(self):
        spec = {"fields": {"npi": {
            "type": "string", "description": "",
            "pattern": r"^\d{10}$",
        }}}
        row = _row(build_model(spec))
        assert row().npi is None
        assert row(npi=None).npi is None

    def test_mismatch_passes_through_unchanged(self):
        """No input-gating: non-matching values are NOT coerced to None."""
        spec = {"fields": {"npi": {
            "type": "string", "description": "",
            "pattern": r"^\d{10}$",
        }}}
        row = _row(build_model(spec))
        # The LLM sees the pattern in the description and should produce a
        # matching canonical form. Non-matching values arriving here are
        # left intact — validation isn't this layer's job.
        assert row(npi="abc").npi == "abc"

    def test_pattern_in_field_description(self):
        """§5.6.1 — pattern surfaced in description so the LLM sees it."""
        spec = {"fields": {"npi": {
            "type": "string", "description": "Box 33a NPI.",
            "pattern": r"^\d{10}$",
        }}}
        desc = _row_props(build_model(spec))["npi"].get("description", "")
        assert "Box 33a NPI." in desc
        assert r"^\d{10}$" in desc

    def test_pattern_in_description_without_user_desc(self):
        """Pattern is surfaced even when no user-provided description."""
        spec = {"fields": {"npi": {
            "type": "string", "pattern": r"^\d{10}$",
        }}}
        desc = _row_props(build_model(spec))["npi"].get("description", "")
        assert r"^\d{10}$" in desc

    def test_pattern_instructs_normalization(self):
        """Pattern hint must tell the LLM to reformat divergent source forms."""
        spec = {"fields": {"ssn": {
            "type": "string",
            "description": "Social Security Number.",
            "pattern": r"^\d{3}-\d{2}-\d{4}$",
        }}}
        desc = _row_props(build_model(spec))["ssn"].get("description", "")
        assert "Social Security Number." in desc
        assert r"^\d{3}-\d{2}-\d{4}$" in desc
        # The hint must explicitly direct the LLM to *normalize* /
        # *reformat* the source value, not just to match.
        assert any(s in desc.lower() for s in ("normaliz", "reformat")), (
            f"Pattern description does not direct the LLM to normalize "
            f"the source value to the canonical form. Got: {desc!r}"
        )
        # And it must distinguish *canonical* form from source form
        assert "canonical" in desc.lower(), (
            f"Pattern description should frame the pattern as the "
            f"canonical output form, not the source form. Got: {desc!r}"
        )

    def test_pattern_on_number_field_augments_description(self):
        """Pattern on `type: number` flows into the description.

        The LLM normalizes "$1,234.56" → "1234.56", Pydantic coerces to float.
        The Pydantic field type stays float; the pattern only shapes what
        the LLM emits as a string before coercion.
        """
        spec = {"fields": {"amount": {
            "type": "number",
            "description": "Dollar amount in box 5.",
            "pattern": r"^\d+(\.\d{2})?$",
        }}}
        model = build_model(spec)
        desc = _row_props(model)["amount"].get("description", "")
        assert "Dollar amount in box 5." in desc
        assert r"^\d+(\.\d{2})?$" in desc
        # Field stays a float — the pattern doesn't change the output type
        from typing import get_args, get_origin
        from types import UnionType
        ann = _row(model).model_fields["amount"].annotation
        if get_origin(ann) is UnionType:
            assert float in get_args(ann)

    def test_pattern_on_category_without_values_treated_as_string(self):
        """category without `values:` compiles to string and honors pattern."""
        spec = {"fields": {"label": {
            "type": "category",
            "description": "Free-text label.",
            "pattern": r"^[A-Z][a-z]+$",
        }}}
        with pytest.warns(UserWarning, match="no `values:`"):
            model = build_model(spec)
        desc = _row_props(model)["label"].get("description", "")
        # User desc + the "infer values" hint + the pattern instruction
        assert "Free-text label." in desc
        assert "infer" in desc.lower()
        assert r"^[A-Z][a-z]+$" in desc

    def test_pattern_silently_dropped_on_boolean(self):
        """Pattern on `type: boolean` is silently dropped at build time.

        validate_blueprint will raise on this, but build_model is permissive
        only when called directly with already-validated dicts; if the
        validate_blueprint gate is bypassed (e.g. internal test
        construction), the pattern just doesn't surface.
        """
        # We have to dodge the gate to land here, so use a string-typed
        # field with pattern alongside to confirm only the boolean's
        # pattern is dropped.
        # (validate_blueprint would reject the boolean+pattern combo, so
        # this tests the build_model branch directly.)
        from petey.schema import build_model as bm
        # Use the public path: validate_blueprint will raise, confirming
        # the gate is on.
        with pytest.raises(ValueError, match="doesn't support a regex"):
            bm({"fields": {"flag": {
                "type": "boolean", "pattern": r"^.+$",
            }}})

    def test_pattern_in_array_child_description(self):
        """Pattern propagates into array row JSON schema, no value gating."""
        spec = {"fields": {"line_items": {
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
        inst = model(items=[{
            "line_items": [{"code": "abc"}],
        }])
        assert inst.items[0].line_items[0].code == "abc"
        # And the row-level field carries the pattern in its description
        schema = _row(model).model_json_schema()
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
            "line_items": {"type": "array"},
            "code": {
                "type": "string", "parent": "line_items",
                "pattern": r"^[A-Z]{3}$",
            },
        }}
        model = build_model(spec)
        inst = model(items=[{
            "line_items": [{"code": "lower"}],
        }])
        assert inst.items[0].line_items[0].code == "lower"

    def test_pattern_with_required_flag(self):
        """`pattern` + `required: true` coexist as orthogonal metadata."""
        spec = {"fields": {"npi": {
            "type": "string", "required": True,
            "pattern": r"^\d{10}$",
        }}}
        row = _row(build_model(spec))
        # Required is metadata; null still allowed in model
        assert row().npi is None
        # Pattern doesn't gate input — non-matching passes through
        assert row(npi="bad").npi == "bad"
        assert row(npi="9876543210").npi == "9876543210"
        # spec dict still carries `required` for downstream
        assert spec["fields"]["npi"]["required"] is True


class TestValidateBlueprint:
    """BPT Spec v1.0 — load-time blueprint validation, now baked into
    build_model so direct callers can't bypass.

    Pattern is allowed on: string, number, category-without-values.
    Rejected on: boolean, date, array, category-with-values.
    """

    def test_valid_blueprint_passes(self):
        from petey.schema import validate_blueprint
        validate_blueprint({"fields": {"npi": {
            "type": "string", "pattern": r"^\d{10}$",
        }}})  # no raise

    def test_pattern_allowed_on_string_and_number(self):
        from petey.schema import validate_blueprint
        validate_blueprint({"fields": {"x": {
            "type": "string", "pattern": r"^\d+$",
        }}})
        validate_blueprint({"fields": {"x": {
            "type": "number", "pattern": r"^\d+(\.\d{2})?$",
        }}})

    def test_pattern_allowed_on_category_without_values(self):
        from petey.schema import validate_blueprint
        validate_blueprint({"fields": {"x": {
            "type": "category", "pattern": r"^.+$",
        }}})

    def test_pattern_rejected_on_unsupported_types(self):
        from petey.schema import validate_blueprint
        for bad_type in ("boolean", "date", "array"):
            spec = {"fields": {"x": {
                "type": bad_type, "pattern": r"^\d+$",
            }}}
            with pytest.raises(ValueError, match="doesn't support"):
                validate_blueprint(spec)

    def test_pattern_rejected_on_category_with_values(self):
        from petey.schema import validate_blueprint
        spec = {"fields": {"x": {
            "type": "category", "values": ["A", "B"],
            "pattern": r"^[AB]$",
        }}}
        with pytest.raises(ValueError, match="doesn't support"):
            validate_blueprint(spec)

    def test_pattern_on_type_aliases_normalized(self):
        from petey.schema import validate_blueprint
        # `cat` with values → category-with-values → raise
        with pytest.raises(ValueError, match="doesn't support"):
            validate_blueprint({"fields": {"x": {
                "type": "cat", "values": ["A", "B"], "pattern": r"^[AB]$",
            }}})
        # `bool` → boolean → raise
        with pytest.raises(ValueError, match="doesn't support"):
            validate_blueprint({"fields": {"x": {
                "type": "bool", "pattern": r"^.+$",
            }}})

    def test_invalid_regex_raises(self):
        from petey.schema import validate_blueprint
        with pytest.raises(ValueError, match="invalid regex"):
            validate_blueprint({"fields": {"x": {
                "type": "string", "pattern": "[unclosed",
            }}})

    def test_recurses_into_array_children(self):
        from petey.schema import validate_blueprint
        # Boolean child with pattern is still rejected
        spec = {"fields": {"items": {
            "type": "array",
            "fields": {
                "flag": {"type": "boolean", "pattern": r"^.+$"},
            },
        }}}
        with pytest.raises(ValueError, match="doesn't support"):
            validate_blueprint(spec)

    def test_array_child_pattern_on_number_allowed(self):
        from petey.schema import validate_blueprint
        spec = {"fields": {"items": {
            "type": "array",
            "fields": {
                "qty": {"type": "number", "pattern": r"^\d+$"},
            },
        }}}
        validate_blueprint(spec)  # no raise

    def test_load_blueprint_invokes_validate(self, tmp_path):
        """A .bpt with a pattern on a forbidden type fails at load."""
        from petey.schema import load_blueprint
        bpt = tmp_path / "bad.bpt"
        bpt.write_text(
            "fields:\n"
            "  flag:\n"
            "    type: boolean\n"
            "    pattern: '^.+$'\n"
        )
        with pytest.raises(ValueError, match="doesn't support"):
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
        # Wrapper has items; row has the npi field
        assert "items" in model.model_fields
        assert "npi" in _row(model).model_fields
        assert spec["fields"]["npi"]["pattern"] == r"^\d{10}$"

    def test_build_model_runs_validate(self):
        """validate_blueprint is baked into build_model — no bypass."""
        spec = {"fields": {"flag": {
            "type": "boolean", "pattern": r"^.+$",
        }}}
        with pytest.raises(ValueError, match="doesn't support"):
            build_model(spec)


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
