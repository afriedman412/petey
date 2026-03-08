"""
Tests for the extraction pipeline using MCI page 1 as test data.
Tests text extraction, schema building, and the full API endpoint.
"""
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


# ---------------------------------------------------------------------------
# Unit tests (no LLM calls)
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_reads_pdf(self):
        from server.extract import extract_text
        text = extract_text(str(MCI_PDF))
        assert "WESTCHESTER COUNTY" in text
        assert "LV910005OM" in text

    def test_contains_all_cases(self):
        from server.extract import extract_text
        text = extract_text(str(MCI_PDF))
        assert "123 VALENTINE LN" in text
        assert "145 TO 147 RIDGE AVE" in text
        assert "153 RIDGE AVE" in text
        assert "149 TO 151 RIDGE AVE" in text
        assert "157 TO 159 RIDGE AVE" in text

    def test_contains_mci_items(self):
        from server.extract import extract_text
        text = extract_text(str(MCI_PDF))
        assert "BALCONY REPLACEMENTS" in text
        assert "INTERIOR STAIRCASE" in text

    def test_total_cases(self):
        from server.extract import extract_text
        text = extract_text(str(MCI_PDF))
        assert "TOTAL CASES:" in text
        assert "5" in text


class TestBuildModel:
    def test_simple_string_fields(self):
        from server.extract import _build_model
        spec = {"fields": {"name": {"type": "string", "description": "A name"}}}
        model = _build_model(spec)
        instance = model(name="test")
        assert instance.name == "test"

    def test_number_field(self):
        from server.extract import _build_model
        spec = {"fields": {"amount": {"type": "number", "description": "Dollar amount"}}}
        model = _build_model(spec)
        instance = model(amount=123.45)
        assert instance.amount == 123.45

    def test_enum_with_values(self):
        from server.extract import _build_model
        spec = {"fields": {"status": {
            "type": "enum",
            "values": ["Open", "Closed"],
            "description": "Status",
        }}}
        model = _build_model(spec)
        schema = model.model_json_schema()
        assert "status_enum" in str(schema)

    def test_enum_without_values_falls_back_to_string(self):
        from server.extract import _build_model
        spec = {"fields": {"status": {"type": "enum", "description": "Status"}}}
        model = _build_model(spec)
        schema = model.model_json_schema()
        # Should be string type, not enum
        assert "status_enum" not in str(schema)
        assert "infer" in str(schema).lower()

    def test_array_record_type(self):
        from server.extract import _build_model
        spec = {
            "record_type": "array",
            "fields": {"address": {"type": "string", "description": "Addr"}},
        }
        model = _build_model(spec)
        schema = model.model_json_schema()
        assert "items" in schema.get("properties", {}) or "items" in schema.get("required", [])

    def test_nested_array_field(self):
        from server.extract import _build_model
        spec = {"fields": {"items": {
            "type": "array",
            "description": "Line items",
            "fields": {
                "name": {"type": "string", "description": "Item name"},
                "cost": {"type": "number", "description": "Cost"},
            },
        }}}
        model = _build_model(spec)
        instance = model(items=[{"name": "Roof", "cost": 100.0}])
        assert len(instance.items) == 1
        assert instance.items[0].name == "Roof"

    def test_mci_schema_builds(self):
        """Build the full MCI schema we'd use for table extraction."""
        from server.extract import _build_model
        spec = {
            "name": "MCI Cases",
            "record_type": "array",
            "fields": {
                "county": {"type": "string", "description": "County name"},
                "address": {"type": "string", "description": "Building address"},
                "docket_number": {"type": "string", "description": "Docket number"},
                "case_status": {"type": "string", "description": "Case status"},
                "closing_date": {"type": "date", "description": "Closing date"},
                "close_code": {"type": "enum", "values": ["GP", "GR", "VO"], "description": "Close code"},
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
        model = _build_model(spec)
        schema = model.model_json_schema()
        # Should be a list wrapper
        assert "items" in schema.get("required", [])


class TestLoadSchema:
    def test_loads_par_schema(self):
        from server.extract import load_schema, SCHEMAS_DIR
        par_path = SCHEMAS_DIR / "par_decision.yaml"
        if not par_path.exists():
            pytest.skip("par_decision.yaml not found")
        model, spec = load_schema(par_path)
        assert spec["name"] == "PAR Decision"
        assert "petitioner" in spec["fields"]


class TestListSchemas:
    def test_lists_yaml_files(self):
        from server.extract import list_schemas, SCHEMAS_DIR
        if not any(SCHEMAS_DIR.glob("*.yaml")):
            pytest.skip("No schemas in schemas/")
        result = list_schemas()
        assert len(result) > 0
        assert "file" in result[0]
        assert "name" in result[0]


# ---------------------------------------------------------------------------
# API tests (no LLM calls)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from httpx import ASGITransport, AsyncClient
    from server.app import app
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_homepage_loads(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "PDF Extractor" in resp.text


@pytest.mark.asyncio
async def test_schemas_endpoint(client):
    resp = await client.get("/schemas")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_parse_yaml_endpoint(client):
    yaml_text = "name: Test\nfields:\n  foo:\n    type: string\n    description: A field"
    resp = await client.post(
        "/parse-yaml",
        json={"yaml": yaml_text},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test"
    assert "foo" in data["fields"]


@pytest.mark.asyncio
async def test_extract_requires_schema(client):
    resp = await client.post(
        "/extract",
        files={"file": ("test.pdf", b"fake", "application/pdf")},
    )
    assert resp.status_code == 400
