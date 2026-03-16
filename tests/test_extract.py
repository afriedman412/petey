"""
Tests for the petey package.
Tests text extraction and schema building using MCI page 1 as test data.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from petey import extract_text, extract_text_pages, build_model, load_schema
from petey.extract import OCR_THRESHOLD

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"
SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_reads_pdf(self):
        text = extract_text(str(MCI_PDF))
        assert "WESTCHESTER COUNTY" in text
        assert "LV910005OM" in text

    def test_contains_all_cases(self):
        text = extract_text(str(MCI_PDF))
        assert "123 VALENTINE LN" in text
        assert "145 TO 147 RIDGE AVE" in text
        assert "153 RIDGE AVE" in text
        assert "149 TO 151 RIDGE AVE" in text
        assert "157 TO 159 RIDGE AVE" in text

    def test_contains_mci_items(self):
        text = extract_text(str(MCI_PDF))
        assert "BALCONY REPLACEMENTS" in text
        assert "INTERIOR STAIRCASE" in text

    def test_total_cases(self):
        text = extract_text(str(MCI_PDF))
        assert "TOTAL CASES:" in text
        assert "5" in text

    def test_extract_text_pages(self):
        pages = extract_text_pages(str(MCI_PDF))
        assert isinstance(pages, list)
        assert len(pages) >= 1
        # Full text should equal pages joined
        full = extract_text(str(MCI_PDF))
        assert full == "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Schema / model building
# ---------------------------------------------------------------------------

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

    def test_array_record_type(self):
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


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

class TestCLI:
    def test_collect_pdfs_from_directory(self):
        from petey.cli import _collect_pdfs
        pdfs = _collect_pdfs([str(FIXTURES)])
        assert len(pdfs) == 1
        assert "mci_page1.pdf" in pdfs[0]

    def test_collect_pdfs_from_file(self):
        from petey.cli import _collect_pdfs
        pdfs = _collect_pdfs([str(MCI_PDF)])
        assert len(pdfs) == 1

    def test_collect_pdfs_empty_dir(self, tmp_path):
        from petey.cli import _collect_pdfs
        pdfs = _collect_pdfs([str(tmp_path)])
        assert pdfs == []

    def test_flatten_simple(self):
        from petey.cli import _flatten
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        flat, keys = _flatten(records)
        assert len(flat) == 2
        assert keys == ["a", "b"]

    def test_flatten_nested(self):
        from petey.cli import _flatten
        records = [{"parent": "x", "children": [{"c": 1}, {"c": 2}]}]
        flat, keys = _flatten(records)
        assert len(flat) == 2
        assert flat[0]["parent"] == "x"
        assert flat[0]["c"] == 1
        assert flat[1]["c"] == 2

    def test_flatten_empty(self):
        from petey.cli import _flatten
        flat, keys = _flatten([])
        assert flat == []
        assert keys == []

    def test_flatten_no_nested(self):
        from petey.cli import _flatten
        records = [{"a": 1}, {"a": 2}]
        flat, keys = _flatten(records)
        assert len(flat) == 2
        assert keys == ["a"]


# ---------------------------------------------------------------------------
# Provider detection & message building
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_openai_model(self):
        from petey.extract import _get_provider
        assert _get_provider("gpt-4.1-mini") == "openai"

    def test_anthropic_model(self):
        from petey.extract import _get_provider
        assert _get_provider("claude-sonnet-4-6") == "anthropic"

    def test_openai_default(self):
        from petey.extract import _get_provider
        assert _get_provider("some-other-model") == "openai"

    def test_litellm_gemini(self):
        from petey.extract import _get_provider
        assert _get_provider("gemini/gemini-2.0-flash") == "litellm"

    def test_litellm_ollama(self):
        from petey.extract import _get_provider
        assert _get_provider("ollama/llama3") == "litellm"

    def test_litellm_bedrock(self):
        from petey.extract import _get_provider
        assert _get_provider("bedrock/anthropic.claude-v2") == "litellm"

    def test_explicit_backend_override(self):
        from petey.extract import _get_provider
        assert _get_provider("gpt-4.1-mini", llm_backend="litellm") == "litellm"
        assert _get_provider("claude-sonnet-4-6", llm_backend="openai") == "openai"

    def test_litellm_client_created(self):
        from petey.extract import _make_client
        client = _make_client("gemini/gemini-2.0-flash")
        assert client is not None


class TestMakeMessages:
    def test_basic_messages(self):
        from petey.extract import _make_messages
        msgs = _make_messages("hello")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "hello" in msgs[1]["content"]

    def test_instructions_appended(self):
        from petey.extract import _make_messages
        msgs = _make_messages("doc text", instructions="Be precise")
        assert "Be precise" in msgs[0]["content"]
        assert "Additional instructions" in msgs[0]["content"]

    def test_no_instructions(self):
        from petey.extract import _make_messages
        msgs = _make_messages("doc text")
        assert "Additional instructions" not in msgs[0]["content"]


# ---------------------------------------------------------------------------
# Schema edge cases
# ---------------------------------------------------------------------------

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
        import re
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
        import re
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


# ---------------------------------------------------------------------------
# Parser backends
# ---------------------------------------------------------------------------

class TestParsers:
    # --- pymupdf (default) ---

    def test_pymupdf_explicit(self):
        text = extract_text(str(MCI_PDF), parser="pymupdf")
        assert "WESTCHESTER COUNTY" in text

    def test_pymupdf_pages_explicit(self):
        pages = extract_text_pages(str(MCI_PDF), parser="pymupdf")
        assert isinstance(pages, list)
        assert len(pages) >= 1
        assert any("WESTCHESTER COUNTY" in p for p in pages)

    # --- tabula ---

    def test_tabula_import_error(self):
        with patch.dict("sys.modules", {"tabula": None}):
            with pytest.raises(ImportError, match="petey\\[tabula\\]"):
                extract_text(str(MCI_PDF), parser="tabula")

    def test_tabula_pages_import_error(self):
        with patch.dict("sys.modules", {"tabula": None}):
            with pytest.raises(ImportError, match="petey\\[tabula\\]"):
                extract_text_pages(str(MCI_PDF), parser="tabula")

    def test_tabula_calls_read_pdf(self):
        mock_tabula = MagicMock()
        # Mock a DataFrame-like object with to_csv
        mock_df = MagicMock()
        mock_df.to_csv.return_value = "col\nval"
        mock_tabula.read_pdf.return_value = [mock_df]
        with patch.dict("sys.modules", {"tabula": mock_tabula}):
            pages = extract_text_pages(str(MCI_PDF), parser="tabula")
        assert isinstance(pages, list)
        assert len(pages) >= 1

    # --- OCR backends ---

    def test_ocr_threshold_constant(self):
        assert OCR_THRESHOLD == 100

    def test_ocr_not_triggered_when_text_present(self):
        # MCI PDF has plenty of text — OCR should never be called
        with patch("petey.extract._ocr_full") as mock_ocr:
            text = extract_text(
                str(MCI_PDF), parser="pymupdf", ocr_fallback=True,
            )
        mock_ocr.assert_not_called()
        assert "WESTCHESTER COUNTY" in text

    def test_ocr_not_triggered_with_ocr_backend(self):
        with patch("petey.extract._ocr_full") as mock_ocr:
            text = extract_text(
                str(MCI_PDF), parser="pymupdf",
                ocr_backend="tesseract",
            )
        mock_ocr.assert_not_called()
        assert "WESTCHESTER COUNTY" in text

    def test_ocr_triggered_when_no_text(self, tmp_path):
        import fitz
        doc = fitz.open()
        doc.new_page()
        blank_pdf = tmp_path / "blank.pdf"
        doc.save(str(blank_pdf))

        with patch(
            "petey.extract._ocr_page", return_value="ocr result"
        ) as mock_ocr:
            text = extract_text(
                str(blank_pdf), parser="pymupdf", ocr_fallback=True,
            )
        mock_ocr.assert_called_once()
        assert "ocr result" in text

    def test_ocr_triggered_with_ocr_backend(self, tmp_path):
        import fitz
        doc = fitz.open()
        doc.new_page()
        blank_pdf = tmp_path / "blank.pdf"
        doc.save(str(blank_pdf))

        with patch(
            "petey.extract._ocr_page", return_value="ocr result"
        ) as mock_ocr:
            text = extract_text(
                str(blank_pdf), parser="pymupdf",
                ocr_backend="tesseract",
            )
        mock_ocr.assert_called_once()
        assert "ocr result" in text

    def test_ocr_not_triggered_when_fallback_disabled(self, tmp_path):
        import fitz
        doc = fitz.open()
        doc.new_page()
        blank_pdf = tmp_path / "blank.pdf"
        doc.save(str(blank_pdf))

        with patch("petey.extract._ocr_page") as mock_ocr:
            extract_text(
                str(blank_pdf), parser="pymupdf", ocr_fallback=False,
            )
        mock_ocr.assert_not_called()

    def test_ocr_backend_resolve_fallback(self):
        from petey.extract import _resolve_ocr_backend
        assert _resolve_ocr_backend(True, "none") == "tesseract"
        assert _resolve_ocr_backend(False, "none") == "none"
        assert _resolve_ocr_backend(True, "mistral") == "mistral"
        assert _resolve_ocr_backend(False, "tesseract") == "tesseract"

    def test_mistral_ocr_import_error(self):
        from petey.extract import _ocr_page_mistral
        mock_page = MagicMock()
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(
                ImportError, match="petey\\[mistral-ocr\\]"
            ):
                _ocr_page_mistral(mock_page)

    def test_mistral_ocr_missing_api_key(self):
        from petey.extract import _ocr_page_mistral
        mock_mistralai = MagicMock()
        mock_page = MagicMock()
        with patch.dict("sys.modules", {"mistralai": mock_mistralai}):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                    _ocr_page_mistral(mock_page)

    def test_ocr_backend_falls_back_to_pdfplumber(self):
        """When --ocr is set with pymupdf parser, should switch to pdfplumber
        to avoid pymupdf4llm's built-in OCR."""
        with patch("petey.extract.pymupdf4llm") as mock_p4l:
            pages = extract_text_pages(
                str(MCI_PDF), parser="pymupdf", ocr_backend="tesseract",
            )
            # pymupdf4llm should NOT be called — pdfplumber is used instead
            mock_p4l.to_markdown.assert_not_called()
            # Should still return page content (from pdfplumber)
            assert isinstance(pages, list)
            assert len(pages) >= 1

    def test_pymupdf_used_when_no_ocr_backend(self):
        """Without --ocr, pymupdf4llm should be used as normal."""
        with patch("petey.extract.pymupdf4llm") as mock_p4l:
            mock_p4l.to_markdown.return_value = [
                {"text": "x" * 200}
            ]
            extract_text_pages(
                str(MCI_PDF), parser="pymupdf", ocr_backend="none",
            )
            mock_p4l.to_markdown.assert_called_once()
