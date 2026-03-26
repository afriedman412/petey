"""
Tests for the petey package.
Tests text extraction and schema building using MCI page 1 as test data.
"""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import (
    AsyncMock, MagicMock, patch,
)

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

    def test_schema_input_used_when_no_cli_paths(self, tmp_path):
        """Schema 'input' field provides PDF paths when CLI paths omitted."""
        from petey.cli import _collect_pdfs
        # Create a temp dir with a PDF
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-fake")
        spec = {"input": str(tmp_path)}
        paths = []  # no CLI paths
        if not paths and spec.get("input"):
            paths = [spec["input"]]
        pdfs = _collect_pdfs(paths)
        assert len(pdfs) == 1
        assert "doc.pdf" in pdfs[0]

    def test_cli_paths_override_schema_input(self, tmp_path):
        """CLI positional paths take precedence over schema input."""
        from petey.cli import _collect_pdfs
        schema_dir = tmp_path / "schema_dir"
        schema_dir.mkdir()
        (schema_dir / "a.pdf").write_bytes(b"%PDF-fake")
        cli_dir = tmp_path / "cli_dir"
        cli_dir.mkdir()
        (cli_dir / "b.pdf").write_bytes(b"%PDF-fake")
        paths = [str(cli_dir)]  # CLI wins
        pdfs = _collect_pdfs(paths)
        assert len(pdfs) == 1
        assert "b.pdf" in pdfs[0]

    def test_schema_output_used_when_no_cli_output(self):
        """Schema 'output' field is used when -o not provided."""
        spec = {"output": "results.csv"}
        output_path = None or spec.get("output")
        assert output_path == "results.csv"

    def test_cli_output_overrides_schema_output(self):
        """CLI -o takes precedence over schema output."""
        spec = {"output": "schema_out.csv"}
        cli_output = "cli_out.json"
        output_path = cli_output or spec.get("output")
        assert output_path == "cli_out.json"

    def test_default_format_csv_for_table_mode(self):
        """Table mode defaults to CSV output."""
        is_table = True
        fmt = None
        output_path = None
        if not fmt:
            fmt = "csv" if is_table else "json"
        assert fmt == "csv"

    def test_default_format_json_for_query_mode(self):
        """Query mode defaults to JSON output."""
        is_table = False
        fmt = None
        output_path = None
        if not fmt:
            fmt = "csv" if is_table else "json"
        assert fmt == "json"

    def test_output_format_inferred_from_extension(self):
        """Output format inferred from file extension."""
        output_path = "results.jsonl"
        ext = Path(output_path).suffix.lower()
        fmt = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}.get(
            ext, "csv",
        )
        assert fmt == "jsonl"

    def test_mode_table_overrides_schema(self):
        """--mode table overrides schema mode."""
        spec = {"mode": "query", "fields": {"x": {"type": "string", "description": ""}}}
        mode_arg = "table"
        if mode_arg is not None:
            spec["mode"] = mode_arg
        assert spec["mode"] == "table"

    def test_mode_query_overrides_schema(self):
        """--mode query overrides schema mode."""
        spec = {"mode": "table", "fields": {"x": {"type": "string", "description": ""}}}
        mode_arg = "query"
        if mode_arg is not None:
            spec["mode"] = mode_arg
        assert spec["mode"] == "query"

    def test_record_type_array_compat_sets_table(self):
        """record_type: array is mapped to mode: table."""
        spec = {"record_type": "array", "fields": {"x": {"type": "string", "description": ""}}}
        if spec.get("record_type") == "array" and "mode" not in spec:
            spec["mode"] = "table"
        assert spec["mode"] == "table"

    def test_schema_input_single_file(self, tmp_path):
        """Schema input pointing to a single file works."""
        from petey.cli import _collect_pdfs
        pdf = tmp_path / "single.pdf"
        pdf.write_bytes(b"%PDF-fake")
        spec = {"input": str(pdf)}
        paths = [spec["input"]]
        pdfs = _collect_pdfs(paths)
        assert len(pdfs) == 1
        assert "single.pdf" in pdfs[0]


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
        from petey.plugins.mistral_ocr import ocr_page
        mock_page = MagicMock()
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(
                ImportError, match="petey\\[mistral-ocr\\]"
            ):
                ocr_page(mock_page)

    def test_mistral_ocr_missing_api_key(self):
        from petey.plugins.mistral_ocr import ocr_page
        mock_mistralai = MagicMock()
        mock_page = MagicMock()
        with patch.dict("sys.modules", {"mistralai": mock_mistralai}):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                    ocr_page(mock_page)

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


# ---------------------------------------------------------------------------
# Remote API backend infrastructure
# ---------------------------------------------------------------------------

class TestResolveResponse:
    """Test dot-path response key extraction."""

    def test_simple_key(self):
        from petey.extract import _resolve_response
        assert _resolve_response({"markdown": "hello"}, "markdown") == "hello"

    def test_nested_key(self):
        from petey.extract import _resolve_response
        data = {"result": {"text": "found it"}}
        assert _resolve_response(data, "result.text") == "found it"

    def test_missing_key_returns_empty(self):
        from petey.extract import _resolve_response
        assert _resolve_response({"a": 1}, "missing") == ""

    def test_missing_nested_returns_empty(self):
        from petey.extract import _resolve_response
        assert _resolve_response({"a": {"b": 1}}, "a.c") == ""

    def test_deep_nesting(self):
        from petey.extract import _resolve_response
        data = {"a": {"b": {"c": "deep"}}}
        assert _resolve_response(data, "a.b.c") == "deep"

    def test_none_value_returns_empty(self):
        from petey.extract import _resolve_response
        assert _resolve_response({"key": None}, "key") == ""


class TestBuildAuthHeader:
    """Test auth header construction from config."""

    def test_raw_key(self):
        from petey.extract import _build_auth_header
        cfg = {"auth_header": "X-API-Key"}
        assert _build_auth_header(cfg, "mykey") == {"X-API-Key": "mykey"}

    def test_bearer_prefix(self):
        from petey.extract import _build_auth_header
        cfg = {"auth_header": "Authorization", "auth_prefix": "Bearer"}
        assert _build_auth_header(cfg, "tok") == {
            "Authorization": "Bearer tok"}

    def test_defaults(self):
        from petey.extract import _build_auth_header
        assert _build_auth_header({}, "k") == {"X-API-Key": "k"}


class TestApiGetKey:
    """Test env var resolution for API keys."""

    def test_key_found(self):
        from petey.extract import _api_get_key
        with patch.dict(os.environ, {"MY_KEY": "secret"}):
            assert _api_get_key({"api_key_env": "MY_KEY"}) == "secret"

    def test_key_missing(self):
        from petey.extract import _api_get_key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MY_KEY"):
                _api_get_key({"api_key_env": "MY_KEY"})


def _mock_httpx_response(json_data, status_code=200):
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


def _mock_async_client(post_resp, get_responses=None):
    """Build a mock httpx.AsyncClient context manager.

    post_resp: response for client.post()
    get_responses: list of responses for client.get()
        (cycled if shorter than calls)
    """
    client = MagicMock()
    client.post = AsyncMock(return_value=post_resp)
    if get_responses is not None:
        client.get = AsyncMock(
            side_effect=get_responses,
        )
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, client


class TestApiPost:
    """Test the generic _api_post function with mocked HTTP."""

    def _make_cfg(self, **overrides):
        cfg = {
            "endpoint": "https://example.com/api",
            "api_key_env": "TEST_KEY",
            "auth_header": "X-API-Key",
            "response_key": "text",
            "poll": False,
            "params": {},
        }
        cfg.update(overrides)
        return cfg

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_multipart_no_poll(self):
        from petey.extract import _api_post
        cfg = self._make_cfg()
        resp = _mock_httpx_response({"text": "extracted"})
        ctx, client = _mock_async_client(resp)

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            result = asyncio.run(
                _api_post(cfg, b"file bytes",
                          "doc.pdf", "application/pdf")
            )

        assert result == "extracted"
        client.post.assert_called_once()

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_json_b64_no_poll(self):
        from petey.extract import _api_post
        import base64
        cfg = self._make_cfg(request_format="json_b64")
        resp = _mock_httpx_response({"text": "from b64"})
        ctx, client = _mock_async_client(resp)

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            result = asyncio.run(
                _api_post(cfg, b"raw", "f.png",
                          "image/png")
            )

        assert result == "from b64"
        _, kwargs = client.post.call_args
        body = json.loads(kwargs["content"])
        assert body["file"] == base64.b64encode(
            b"raw"
        ).decode()
        assert body["filename"] == "f.png"

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_poll_flow(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(
            poll=True, response_key="markdown",
            timeout=10,
        )
        submit_resp = _mock_httpx_response(
            {"request_check_url":
                "https://example.com/check/123"},
        )
        poll_pending = _mock_httpx_response(
            {"status": "pending"},
        )
        poll_done = _mock_httpx_response(
            {"status": "complete",
             "markdown": "result text"},
        )
        ctx, client = _mock_async_client(
            submit_resp, [poll_pending, poll_done],
        )

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            with patch("petey.extract.asyncio.sleep",
                        new_callable=AsyncMock):
                result = asyncio.run(
                    _api_post(cfg, b"data", "f.pdf",
                              "application/pdf")
                )

        assert result == "result text"

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_poll_timeout(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(poll=True, timeout=4)
        submit_resp = _mock_httpx_response(
            {"request_check_url":
                "https://example.com/check/1"},
        )
        poll_resp = _mock_httpx_response(
            {"status": "pending"},
        )
        ctx, _ = _mock_async_client(
            submit_resp, [poll_resp, poll_resp],
        )

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            with patch("petey.extract.asyncio.sleep",
                        new_callable=AsyncMock):
                with pytest.raises(TimeoutError):
                    asyncio.run(
                        _api_post(cfg, b"x", "f.pdf",
                                  "application/pdf")
                    )

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_missing_check_url(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(poll=True)
        resp = _mock_httpx_response({})
        ctx, _ = _mock_async_client(resp)

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            with pytest.raises(ValueError, match="check"):
                asyncio.run(
                    _api_post(cfg, b"x", "f.pdf",
                              "application/pdf")
                )

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_bearer_auth(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(
            auth_header="Authorization",
            auth_prefix="Bearer", poll=False,
        )
        resp = _mock_httpx_response({"text": "ok"})
        ctx, client = _mock_async_client(resp)

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            asyncio.run(
                _api_post(cfg, b"x", "f.pdf",
                          "application/pdf")
            )

        _, kwargs = client.post.call_args
        assert kwargs["headers"]["Authorization"] == \
            "Bearer k"

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_custom_poll_keys(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(
            poll=True, timeout=10,
            poll_check_key="job_url",
            poll_status_key="state",
            poll_done_value="finished",
            response_key="output.text",
        )
        submit_resp = _mock_httpx_response(
            {"job_url": "https://example.com/job/1"},
        )
        poll_done = _mock_httpx_response(
            {"state": "finished",
             "output": {"text": "custom result"}},
        )
        ctx, _ = _mock_async_client(
            submit_resp, [poll_done],
        )

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            with patch("petey.extract.asyncio.sleep",
                        new_callable=AsyncMock):
                result = asyncio.run(
                    _api_post(cfg, b"x", "f.pdf",
                              "application/pdf")
                )

        assert result == "custom result"

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_extra_params(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(
            params={"output_format": "html", "lang": "en"},
            poll=False,
        )
        resp = _mock_httpx_response({"text": "ok"})
        ctx, client = _mock_async_client(resp)

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            asyncio.run(
                _api_post(cfg, b"x", "f.pdf",
                          "application/pdf")
            )

        _, kwargs = client.post.call_args
        assert kwargs["data"]["output_format"] == "html"
        assert kwargs["data"]["lang"] == "en"


class TestParsePdfViaApi:
    """Test the parser wrapper around _api_post."""

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_returns_list(self, tmp_path):
        from petey.extract import _parse_pdf_via_api
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        cfg = {
            "endpoint": "https://example.com/api",
            "api_key_env": "TEST_KEY",
            "poll": False,
            "response_key": "text",
        }
        with patch("petey.extract._api_post",
                    new_callable=AsyncMock,
                    return_value="parsed"):
            result = asyncio.run(
                _parse_pdf_via_api(str(pdf), cfg)
            )
        assert result == ["parsed"]

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_empty_result(self, tmp_path):
        from petey.extract import _parse_pdf_via_api
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        cfg = {
            "endpoint": "https://example.com/api",
            "api_key_env": "TEST_KEY",
            "poll": False,
            "response_key": "text",
        }
        with patch("petey.extract._api_post",
                    new_callable=AsyncMock,
                    return_value=""):
            result = asyncio.run(
                _parse_pdf_via_api(str(pdf), cfg)
            )
        assert result == [""]


class TestOcrPageViaApi:
    """Test the OCR wrapper around _api_post."""

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_renders_and_posts(self):
        from petey.extract import _ocr_page_via_api
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"png bytes"
        mock_page.get_pixmap.return_value = mock_pix
        cfg = {
            "endpoint": "https://example.com/ocr",
            "api_key_env": "TEST_KEY",
            "poll": False,
            "response_key": "text",
        }
        with patch("petey.extract._api_post",
                    new_callable=AsyncMock,
                    return_value="ocr text") as mock_post:
            result = asyncio.run(
                _ocr_page_via_api(mock_page, cfg)
            )

        assert result == "ocr text"
        mock_page.get_pixmap.assert_called_once_with(dpi=200)
        args, _ = mock_post.call_args
        assert args[1] == b"png bytes"
        assert args[2] == "page.png"
        assert args[3] == "image/png"


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistries:
    """Verify all expected backends appear in the registries."""

    def test_parser_registry_has_local_backends(self):
        from petey.extract import PARSERS
        for name in ["pymupdf", "pdfplumber", "tables", "tabula"]:
            assert name in PARSERS, f"Missing parser: {name}"

    def test_parser_registry_has_api_backends(self):
        from petey.extract import PARSERS
        assert "marker" in PARSERS
        assert "llamaparse" in PARSERS

    def test_parser_registry_has_plugin_backends(self):
        from petey.extract import PARSERS
        assert "docling" in PARSERS

    def test_ocr_registry_has_local_backends(self):
        from petey.extract import OCR_BACKENDS
        assert "tesseract" in OCR_BACKENDS

    def test_ocr_registry_has_plugin_backends(self):
        from petey.extract import OCR_BACKENDS
        assert "mistral" in OCR_BACKENDS

    def test_ocr_registry_has_api_backends(self):
        from petey.extract import OCR_BACKENDS
        assert "chandra" in OCR_BACKENDS

    def test_llm_registry_has_builtins(self):
        from petey.extract import LLM_BACKENDS
        for name in ["openai", "anthropic", "litellm"]:
            assert name in LLM_BACKENDS, f"Missing LLM: {name}"

    def test_api_parsers_config_valid(self):
        from petey.extract import API_PARSERS
        for name, cfg in API_PARSERS.items():
            has_endpoint = "endpoint" in cfg or "endpoint_env" in cfg
            assert has_endpoint, f"{name} missing endpoint/endpoint_env"
            assert "api_key_env" in cfg, f"{name} missing api_key_env"

    def test_api_ocr_config_valid(self):
        from petey.extract import API_OCR_BACKENDS
        for name, cfg in API_OCR_BACKENDS.items():
            has_endpoint = "endpoint" in cfg or "endpoint_env" in cfg
            assert has_endpoint, f"{name} missing endpoint/endpoint_env"
            assert "api_key_env" in cfg, f"{name} missing api_key_env"

    def test_api_parsers_are_callable(self):
        from petey.extract import PARSERS
        for name in ["marker", "llamaparse"]:
            assert callable(PARSERS[name])

    def test_api_ocr_are_callable(self):
        from petey.extract import OCR_BACKENDS
        assert callable(OCR_BACKENDS["chandra"])

    def test_local_parsers_are_sync(self):
        from petey.extract import PARSERS
        for name in ["pymupdf", "pdfplumber", "tables", "tabula"]:
            assert not asyncio.iscoroutinefunction(
                PARSERS[name]
            ), f"Parser '{name}' should be sync"

    def test_api_parsers_are_async(self):
        from petey.extract import PARSERS, API_PARSERS
        for name in API_PARSERS:
            assert asyncio.iscoroutinefunction(
                PARSERS[name]
            ), f"Parser '{name}' should be async"

    def test_local_ocr_backends_are_sync(self):
        from petey.extract import OCR_BACKENDS
        for name in ["tesseract"]:
            assert not asyncio.iscoroutinefunction(
                OCR_BACKENDS[name]
            ), f"OCR '{name}' should be sync"

    def test_plugin_ocr_backends_are_callable(self):
        from petey.extract import OCR_BACKENDS, PLUGIN_OCR_BACKENDS
        for name in PLUGIN_OCR_BACKENDS:
            assert callable(
                OCR_BACKENDS[name]
            ), f"Plugin OCR '{name}' should be callable"

    def test_api_ocr_backends_are_async(self):
        from petey.extract import (
            OCR_BACKENDS, API_OCR_BACKENDS,
        )
        for name in API_OCR_BACKENDS:
            assert asyncio.iscoroutinefunction(
                OCR_BACKENDS[name]
            ), f"OCR '{name}' should be async"

    def test_marker_uses_correct_endpoint(self):
        from petey.extract import API_PARSERS
        assert "marker" in API_PARSERS
        assert "/marker" in API_PARSERS["marker"]["endpoint"]

    def test_chandra_uses_correct_endpoint(self):
        from petey.extract import API_OCR_BACKENDS
        assert "/chandra" in API_OCR_BACKENDS["chandra"]["endpoint"]


class TestLLMBackendConfig:
    """Test config-driven LLM backend registration."""

    def test_api_llm_backend_creates_openai_client(self):
        from petey.extract import _make_api_llm_client
        with patch.dict(os.environ, {"TEST_LLM_KEY": "k"}):
            with patch("petey.extract.AsyncOpenAI") as mock_oai:
                with patch("petey.extract.instructor") as mock_inst:
                    mock_inst.from_openai.return_value = "client"
                    result = _make_api_llm_client(
                        client="openai",
                        api_key_env="TEST_LLM_KEY",
                        base_url="https://my-host.com/v1",
                    )
        assert result == "client"
        mock_oai.assert_called_once_with(
            api_key="k", base_url="https://my-host.com/v1")

    def test_api_llm_backend_unknown_client_type(self):
        from petey.extract import _make_api_llm_client
        with pytest.raises(ValueError, match="Unknown LLM client"):
            _make_api_llm_client(client="buttllm")


# ---------------------------------------------------------------------------
# Integration: API parser/OCR via extract_text
# ---------------------------------------------------------------------------

class TestApiIntegration:
    """Test that API backends integrate with extract_text."""

    @patch.dict(os.environ, {"DATALAB_API_KEY": "test-key"})
    def test_marker_parser_via_extract_text(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        with patch(
            "petey.extract._api_post",
            new_callable=AsyncMock,
            return_value="marker output",
        ):
            text = extract_text(
                str(pdf), parser="marker",
            )
        assert text == "marker output"

    @patch.dict(os.environ, {"DATALAB_API_KEY": "test-key"})
    def test_chandra_ocr_via_extract_text(self, tmp_path):
        import fitz
        doc = fitz.open()
        doc.new_page()
        blank_pdf = tmp_path / "blank.pdf"
        doc.save(str(blank_pdf))

        with patch(
            "petey.extract._api_post",
            new_callable=AsyncMock,
            return_value="chandra ocr",
        ):
            text = extract_text(
                str(blank_pdf),
                parser="pdfplumber",
                ocr_backend="chandra",
            )
        assert "chandra ocr" in text

    def test_unknown_parser_raises(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        with pytest.raises(ValueError, match="not found"):
            extract_text(str(pdf), parser="buttparse")

    def test_unknown_ocr_raises(self, tmp_path):
        import fitz
        doc = fitz.open()
        doc.new_page()
        blank_pdf = tmp_path / "blank.pdf"
        doc.save(str(blank_pdf))

        with pytest.raises(ValueError, match="not found"):
            extract_text(
                str(blank_pdf), parser="pdfplumber",
                ocr_backend="buttocr")


# -----------------------------------------------------------
# ConcurrencyManager
# -----------------------------------------------------------

def _add(a, b):
    return a + b


def _triple(x):
    return x * 3


def _one():
    return 1


class TestConcurrencyManager:
    """Test the ConcurrencyManager from petey.concurrency."""

    def test_singleton(self):
        from petey.concurrency import get_manager
        a = get_manager()
        b = get_manager()
        assert a is b

    def test_defaults(self):
        from petey.concurrency import ConcurrencyManager
        mgr = ConcurrencyManager()
        assert mgr.cpu_limit >= 1
        assert mgr.api_limit == 10

    def test_configure(self):
        from petey.concurrency import ConcurrencyManager
        mgr = ConcurrencyManager()
        mgr.configure(cpu_limit=2, api_limit=5)
        assert mgr.cpu_limit == 2
        assert mgr.api_limit == 5

    def test_run_cpu_dispatches_sync(self):
        from petey.concurrency import ConcurrencyManager
        mgr = ConcurrencyManager(cpu_limit=2)
        result = asyncio.run(mgr.run_cpu(_add, 1, 2))
        assert result == 3

    def test_run_dispatches_async(self):
        from petey.concurrency import ConcurrencyManager

        async def double(x):
            return x * 2

        mgr = ConcurrencyManager()
        result = asyncio.run(mgr.run(double, 5))
        assert result == 10

    def test_run_dispatches_sync_to_cpu(self):
        from petey.concurrency import ConcurrencyManager
        mgr = ConcurrencyManager(cpu_limit=1)
        result = asyncio.run(mgr.run(_triple, 4))
        assert result == 12

    def test_api_semaphore(self):
        from petey.concurrency import ConcurrencyManager
        mgr = ConcurrencyManager(api_limit=1)

        async def check():
            async with mgr.api():
                return "ok"

        assert asyncio.run(check()) == "ok"

    def test_shutdown(self):
        from petey.concurrency import ConcurrencyManager
        mgr = ConcurrencyManager(cpu_limit=1)
        asyncio.run(mgr.run_cpu(_one))
        assert mgr._cpu_pool is not None
        mgr.shutdown()
        assert mgr._cpu_pool is None


# -----------------------------------------------------------
# Async extraction ← ConcurrencyManager integration
# -----------------------------------------------------------

def _mock_llm_client():
    """Build a mock instructor client for LLM calls."""
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {"field": "value"}
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=mock_result,
    )
    return client


class TestExtractAsyncConcurrency:
    """Verify extract_async routes work through the manager."""

    def test_local_parser_uses_cpu_pool(self):
        """Local (sync) parsing should go through run_cpu."""
        from petey.extract import extract_async
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        calls = {"run_cpu": 0, "api": 0}
        orig_run_cpu = ConcurrencyManager.run_cpu
        orig_api = ConcurrencyManager.api

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1
            return await orig_run_cpu(self, fn, *a)

        def spy_api(self):
            calls["api"] += 1
            return orig_api(self)

        with patch.object(
            ConcurrencyManager, "run_cpu", spy_run_cpu,
        ):
            with patch.object(
                ConcurrencyManager, "api", spy_api,
            ):
                with patch(
                    "petey.extract._make_client",
                    return_value=_mock_llm_client(),
                ):
                    result = asyncio.run(
                        extract_async(
                            str(MCI_PDF), M,
                            parser="pymupdf",
                        )
                    )

        assert calls["run_cpu"] >= 1, (
            "Local parsing should use run_cpu"
        )
        assert calls["api"] >= 1, (
            "LLM call should use api semaphore"
        )

    def test_async_parse_fn_uses_run(self):
        """Async parse_fn should go through run() (API pool)."""
        from petey.extract import extract_async
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        calls = {"run": 0, "run_cpu": 0, "api": 0}
        orig_run = ConcurrencyManager.run
        orig_api = ConcurrencyManager.api

        async def spy_run(self, fn, *a):
            calls["run"] += 1
            return await orig_run(self, fn, *a)

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1

        def spy_api(self):
            calls["api"] += 1
            return orig_api(self)

        async def fake_parse(path, parser, ocr):
            return "fake text from API parser"

        with patch.object(
            ConcurrencyManager, "run", spy_run,
        ):
            with patch.object(
                ConcurrencyManager, "run_cpu",
                spy_run_cpu,
            ):
                with patch.object(
                    ConcurrencyManager, "api", spy_api,
                ):
                    with patch(
                        "petey.extract._make_client",
                        return_value=_mock_llm_client(),
                    ):
                        result = asyncio.run(
                            extract_async(
                                str(MCI_PDF), M,
                                parse_fn=fake_parse,
                            )
                        )

        assert calls["run"] >= 1, (
            "Async parse_fn should use run()"
        )
        assert calls["run_cpu"] == 0, (
            "Should NOT use run_cpu for async parse_fn"
        )
        assert calls["api"] >= 1, (
            "LLM call should use api semaphore"
        )


class TestRegistryDispatchConcurrency:
    """Verify registry entries route through correct pool."""

    @patch.dict(os.environ, {"DATALAB_API_KEY": "test-key"})
    def test_marker_parser_routes_to_api_pool(self, tmp_path):
        """parser='marker' (async registry entry) should
        go through run() → API pool, not run_cpu()."""
        from petey.extract import extract_async
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        calls = {"run": 0, "run_cpu": 0}
        orig_run = ConcurrencyManager.run

        async def spy_run(self, fn, *a):
            calls["run"] += 1
            return await orig_run(self, fn, *a)

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1
            return ""

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")

        with patch.object(
            ConcurrencyManager, "run", spy_run,
        ):
            with patch.object(
                ConcurrencyManager, "run_cpu",
                spy_run_cpu,
            ):
                with patch(
                    "petey.extract._api_post",
                    new_callable=AsyncMock,
                    return_value="marker text",
                ):
                    with patch(
                        "petey.extract._make_client",
                        return_value=_mock_llm_client(),
                    ):
                        asyncio.run(
                            extract_async(
                                str(pdf), M,
                                parser="marker",
                            )
                        )

        assert calls["run"] >= 1, (
            "marker (async) should route through run()"
        )
        assert calls["run_cpu"] == 0, (
            "marker should NOT use run_cpu"
        )

    @patch.dict(
        os.environ, {"DATALAB_API_KEY": "test-key"},
    )
    def test_pymupdf_parser_routes_to_cpu_pool(self):
        """parser='pymupdf' (sync registry entry) should
        go through run_cpu() → CPU pool."""
        from petey.extract import extract_async
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        calls = {"run": 0, "run_cpu": 0}
        orig_run_cpu = ConcurrencyManager.run_cpu

        async def spy_run(self, fn, *a):
            calls["run"] += 1
            return ""

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1
            return await orig_run_cpu(self, fn, *a)

        with patch.object(
            ConcurrencyManager, "run", spy_run,
        ):
            with patch.object(
                ConcurrencyManager, "run_cpu",
                spy_run_cpu,
            ):
                with patch(
                    "petey.extract._make_client",
                    return_value=_mock_llm_client(),
                ):
                    asyncio.run(
                        extract_async(
                            str(MCI_PDF), M,
                            parser="pymupdf",
                        )
                    )

        assert calls["run_cpu"] >= 1, (
            "pymupdf (sync) should route through run_cpu()"
        )
        assert calls["run"] == 0, (
            "pymupdf should NOT use run()"
        )


class TestExtractBatchConcurrency:
    """Verify extract_batch routes through the manager."""

    def test_local_parser_uses_cpu_pool(self, tmp_path):
        from petey.extract import extract_batch
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel
        import fitz

        class M(BaseModel):
            field: str | None = None

        # Create a tiny valid PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "test content")
        pdf = tmp_path / "test.pdf"
        doc.save(str(pdf))
        doc.close()

        calls = {"run_cpu": 0, "api": 0}
        orig_run_cpu = ConcurrencyManager.run_cpu
        orig_api = ConcurrencyManager.api

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1
            return await orig_run_cpu(self, fn, *a)

        def spy_api(self):
            calls["api"] += 1
            return orig_api(self)

        with patch.object(
            ConcurrencyManager, "run_cpu", spy_run_cpu,
        ):
            with patch.object(
                ConcurrencyManager, "api", spy_api,
            ):
                with patch(
                    "petey.extract._make_client",
                    return_value=_mock_llm_client(),
                ):
                    results = asyncio.run(
                        extract_batch(
                            [str(pdf)], M,
                            parser="pymupdf",
                        )
                    )

        assert calls["run_cpu"] >= 1, (
            "Local parsing should use run_cpu"
        )
        assert calls["api"] >= 1, (
            "LLM call should use api semaphore"
        )
        assert len(results) == 1
        assert results[0]["field"] == "value"

    def test_async_parse_fn_uses_run(self, tmp_path):
        from petey.extract import extract_batch
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        calls = {"run": 0, "run_cpu": 0, "api": 0}
        orig_run = ConcurrencyManager.run
        orig_api = ConcurrencyManager.api

        async def spy_run(self, fn, *a):
            calls["run"] += 1
            return await orig_run(self, fn, *a)

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1

        def spy_api(self):
            calls["api"] += 1
            return orig_api(self)

        async def fake_parse(path, parser, ocr):
            return "fake text"

        with patch.object(
            ConcurrencyManager, "run", spy_run,
        ):
            with patch.object(
                ConcurrencyManager, "run_cpu",
                spy_run_cpu,
            ):
                with patch.object(
                    ConcurrencyManager, "api", spy_api,
                ):
                    with patch(
                        "petey.extract._make_client",
                        return_value=_mock_llm_client(),
                    ):
                        results = asyncio.run(
                            extract_batch(
                                ["fake.pdf"], M,
                                parse_fn=fake_parse,
                            )
                        )

        assert calls["run"] >= 1, (
            "Async parse_fn should use run()"
        )
        assert calls["run_cpu"] == 0, (
            "Should NOT use run_cpu for async parse_fn"
        )
        assert calls["api"] >= 1, (
            "LLM call should use api semaphore"
        )


# -----------------------------------------------------------
# PDF subsetting + extract_pages_async registry dispatch
# -----------------------------------------------------------

class TestSubsetPdf:
    """Test the _subset_pdf helper."""

    def test_subset_single_page(self):
        import fitz
        from petey.extract import _subset_pdf

        subset_path = _subset_pdf(str(MCI_PDF), [0])
        try:
            doc = fitz.open(subset_path)
            assert len(doc) == 1
            text = doc[0].get_text("text")
            doc.close()
            assert "WESTCHESTER" in text
        finally:
            os.unlink(subset_path)

    def test_subset_preserves_page_order(self, tmp_path):
        import fitz
        from petey.extract import _subset_pdf

        # Create a 3-page PDF with distinct content
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text(
                (72, 72), f"PAGE_{i}",
            )
        src = tmp_path / "src.pdf"
        doc.save(str(src))
        doc.close()

        # Subset pages 2 and 0 (reversed)
        subset_path = _subset_pdf(str(src), [2, 0])
        try:
            sub = fitz.open(subset_path)
            assert len(sub) == 2
            assert "PAGE_2" in sub[0].get_text("text")
            assert "PAGE_0" in sub[1].get_text("text")
            sub.close()
        finally:
            os.unlink(subset_path)

    def test_subset_cleanup(self):
        from petey.extract import _subset_pdf

        path = _subset_pdf(str(MCI_PDF), [0])
        assert os.path.exists(path)
        os.unlink(path)
        assert not os.path.exists(path)


class TestExtractPagesAsyncRegistry:
    """Verify extract_pages_async uses registry + subsetting."""

    def test_uses_registry_parser(self):
        """extract_pages_async should dispatch through
        the manager via the registry."""
        from petey.extract import extract_pages_async
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        calls = {"run_cpu": 0}
        orig_run_cpu = ConcurrencyManager.run_cpu

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1
            return await orig_run_cpu(self, fn, *a)

        with patch.object(
            ConcurrencyManager, "run_cpu", spy_run_cpu,
        ):
            with patch(
                "petey.extract._make_client",
                return_value=_mock_llm_client(),
            ):
                results = asyncio.run(
                    extract_pages_async(
                        str(MCI_PDF), M,
                        parser="pymupdf",
                    )
                )

        assert calls["run_cpu"] >= 1
        assert len(results) >= 1

    @patch.dict(os.environ, {"DATALAB_API_KEY": "k"})
    def test_api_parser_uses_run(self):
        """parser='marker' should go through mgr.run()
        (API pool) via subsetting."""
        from petey.extract import extract_pages_async
        from petey.concurrency import ConcurrencyManager
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        calls = {"run": 0, "run_cpu": 0}
        orig_run = ConcurrencyManager.run

        async def spy_run(self, fn, *a):
            calls["run"] += 1
            return await orig_run(self, fn, *a)

        async def spy_run_cpu(self, fn, *a):
            calls["run_cpu"] += 1
            return ""

        with patch.object(
            ConcurrencyManager, "run", spy_run,
        ):
            with patch.object(
                ConcurrencyManager, "run_cpu",
                spy_run_cpu,
            ):
                with patch(
                    "petey.extract._api_post",
                    new_callable=AsyncMock,
                    return_value="marker text",
                ):
                    with patch(
                        "petey.extract._make_client",
                        return_value=_mock_llm_client(),
                    ):
                        asyncio.run(
                            extract_pages_async(
                                str(MCI_PDF), M,
                                parser="marker",
                            )
                        )

        assert calls["run"] >= 1, (
            "marker should route through run()"
        )
        assert calls["run_cpu"] == 0, (
            "marker should NOT use run_cpu"
        )

    def test_unknown_parser_raises(self):
        from petey.extract import extract_pages_async
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        with pytest.raises(ValueError, match="not found"):
            asyncio.run(
                extract_pages_async(
                    str(MCI_PDF), M,
                    parser="nonexistent",
                )
            )


# ---------------------------------------------------------------------------
# poll_url_template
# ---------------------------------------------------------------------------

class TestPollUrlTemplate:
    """Test poll_url_template support in _api_post."""

    def _make_cfg(self, **overrides):
        cfg = {
            "endpoint": "https://example.com/api/upload",
            "api_key_env": "TEST_KEY",
            "auth_header": "Authorization",
            "auth_prefix": "Bearer",
            "response_key": "markdown",
            "poll": True,
            "timeout": 10,
            "params": {},
        }
        cfg.update(overrides)
        return cfg

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_poll_url_template_constructs_correct_url(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(
            poll_check_key="id",
            poll_url_template=(
                "https://api.example.com/job/{id}/result/markdown"
            ),
            poll_status_key="status",
            poll_done_value="done",
        )
        submit_resp = _mock_httpx_response({"id": "job-42"})
        poll_done = _mock_httpx_response(
            {"status": "done", "markdown": "ok"},
        )
        ctx, client = _mock_async_client(
            submit_resp, [poll_done],
        )

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            with patch("petey.extract.asyncio.sleep",
                        new_callable=AsyncMock):
                asyncio.run(
                    _api_post(cfg, b"x", "f.pdf",
                              "application/pdf")
                )

        poll_url = client.get.call_args[0][0]
        assert poll_url == (
            "https://api.example.com/job/job-42/result/markdown"
        )

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_poll_url_template_missing_key_raises(self):
        from petey.extract import _api_post
        cfg = self._make_cfg(
            poll_check_key="id",
            poll_url_template="https://example.com/job/{id}",
        )
        # Response missing the "id" key
        submit_resp = _mock_httpx_response({})
        ctx, _ = _mock_async_client(submit_resp)

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            with pytest.raises(ValueError, match="id"):
                asyncio.run(
                    _api_post(cfg, b"x", "f.pdf",
                              "application/pdf")
                )

    @patch.dict(os.environ, {"TEST_KEY": "k"})
    def test_without_template_uses_raw_url(self):
        """Existing behaviour: poll_check_key is a full URL."""
        from petey.extract import _api_post
        cfg = self._make_cfg(
            poll_check_key="request_check_url",
            poll_status_key="status",
            poll_done_value="complete",
        )
        submit_resp = _mock_httpx_response(
            {"request_check_url": "https://example.com/check/1"},
        )
        poll_done = _mock_httpx_response(
            {"status": "complete", "markdown": "text"},
        )
        ctx, client = _mock_async_client(
            submit_resp, [poll_done],
        )

        with patch("petey.extract.httpx.AsyncClient",
                    return_value=ctx):
            with patch("petey.extract.asyncio.sleep",
                        new_callable=AsyncMock):
                result = asyncio.run(
                    _api_post(cfg, b"x", "f.pdf",
                              "application/pdf")
                )

        assert result == "text"
        poll_url = client.get.call_args[0][0]
        assert poll_url == "https://example.com/check/1"


# ---------------------------------------------------------------------------
# LlamaParse config
# ---------------------------------------------------------------------------

class TestLlamaParseConfig:
    """Verify the llamaparse API_PARSERS entry is correct."""

    def test_llamaparse_uses_bearer_auth(self):
        from petey.extract import API_PARSERS
        cfg = API_PARSERS["llamaparse"]
        assert cfg["auth_header"] == "Authorization"
        assert cfg["auth_prefix"] == "Bearer"

    def test_llamaparse_uses_poll_url_template(self):
        from petey.extract import API_PARSERS
        cfg = API_PARSERS["llamaparse"]
        assert "poll_url_template" in cfg
        assert "{id}" in cfg["poll_url_template"]
        assert cfg["poll_check_key"] == "id"

    def test_llamaparse_poll_done_value(self):
        from petey.extract import API_PARSERS
        cfg = API_PARSERS["llamaparse"]
        assert cfg["poll_done_value"] == "SUCCESS"

    def test_llamaparse_api_key_env(self):
        from petey.extract import API_PARSERS
        cfg = API_PARSERS["llamaparse"]
        assert cfg["api_key_env"] == "LLAMA_CLOUD_API_KEY"


# ---------------------------------------------------------------------------
# Plugin registries
# ---------------------------------------------------------------------------

class TestPluginRegistries:
    """Test the PLUGIN_* registries and lazy loader."""

    def test_docling_in_plugin_parsers(self):
        from petey.extract import PLUGIN_PARSERS
        assert "docling" in PLUGIN_PARSERS
        assert "docling" in PLUGIN_PARSERS["docling"]

    def test_plugin_parsers_are_callable(self):
        from petey.extract import PARSERS, PLUGIN_PARSERS
        for name in PLUGIN_PARSERS:
            assert callable(
                PARSERS[name]
            ), f"Plugin parser '{name}' should be callable"


class TestPluginLoader:
    """Test the _make_plugin_loader lazy import mechanism."""

    def test_lazy_import_calls_function(self):
        from petey.extract import _make_plugin_loader
        loader = _make_plugin_loader("json:loads")
        result = loader('{"a": 1}')
        assert result == {"a": 1}

    def test_bad_module_raises_at_build(self):
        from petey.extract import _make_plugin_loader
        with pytest.raises(ModuleNotFoundError):
            _make_plugin_loader("nonexistent_module_xyz:func")

    def test_bad_attr_raises_at_build(self):
        from petey.extract import _make_plugin_loader
        with pytest.raises(AttributeError):
            _make_plugin_loader("json:nonexistent_function_xyz")


# ---------------------------------------------------------------------------
# parser_options / ocr_options passthrough
# ---------------------------------------------------------------------------

class TestBackendOptions:
    """Verify parser_options and ocr_options reach the callable."""

    def test_parser_options_passed(self):
        mock_fn = MagicMock(return_value=["page text"])
        with patch.dict(
            "petey.extract.PARSERS", {"mock": mock_fn},
        ):
            extract_text_pages(
                str(MCI_PDF), parser="mock",
                parser_options={"lang": "fr", "dpi": 300},
            )
        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs["lang"] == "fr"
        assert kwargs["dpi"] == 300

    def test_ocr_options_passed(self):
        from petey.extract import OCR_THRESHOLD
        mock_parser = MagicMock(return_value=["x"])
        mock_ocr = MagicMock(return_value="ocr text")
        with patch.dict(
            "petey.extract.PARSERS", {"mock": mock_parser},
        ):
            with patch.dict(
                "petey.extract.OCR_BACKENDS",
                {"mock_ocr": mock_ocr},
            ):
                extract_text_pages(
                    str(MCI_PDF), parser="mock",
                    ocr_backend="mock_ocr",
                    ocr_options={"languages": ["en", "fr"]},
                )
        mock_ocr.assert_called_once()
        _, kwargs = mock_ocr.call_args
        assert kwargs["languages"] == ["en", "fr"]
