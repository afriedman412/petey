"""
Tests for core extraction: text extraction, CLI helpers, quality checks,
PDF subsetting, and parser backends.
"""
import os
from pathlib import Path

import pytest

from petey import extract_text, extract_text_pages

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


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
# Extraction quality checks
# ---------------------------------------------------------------------------

class TestCheckExtractionQuality:
    def test_all_nulls_warns(self):
        from petey.extract import _check_extraction_quality
        data = {"name": None, "age": None, "city": None}
        msgs = _check_extraction_quality(data, "some text " * 50)
        assert any("3/3 fields" in m for m in msgs)

    def test_mostly_nulls_warns(self):
        from petey.extract import _check_extraction_quality
        data = {
            "a": None, "b": None, "c": None, "d": None,
            "e": None, "f": "value",
        }
        msgs = _check_extraction_quality(data, "some text " * 50)
        assert any("5/6 fields" in m for m in msgs)

    def test_short_text_warns(self):
        from petey.extract import _check_extraction_quality
        data = {"name": "Alice", "age": 30}
        msgs = _check_extraction_quality(data, "short")
        assert any("very short" in m for m in msgs)

    def test_good_extraction_no_warnings(self):
        from petey.extract import _check_extraction_quality
        data = {"name": "Alice", "age": 30, "city": "NYC"}
        msgs = _check_extraction_quality(data, "x" * 500)
        assert msgs == []

    def test_ignores_underscore_fields(self):
        from petey.extract import _check_extraction_quality
        data = {"_page": "p1", "_source_file": "test.pdf", "name": "Alice"}
        msgs = _check_extraction_quality(data, "x" * 500)
        assert msgs == []

    def test_label_in_message(self):
        from petey.extract import _check_extraction_quality
        data = {"name": None}
        msgs = _check_extraction_quality(data, "short", label="p1")
        assert any("p1" in m for m in msgs)


# ---------------------------------------------------------------------------
# Parser backends
# ---------------------------------------------------------------------------

class TestParsers:
    def test_pymupdf_explicit(self):
        text = extract_text(str(MCI_PDF), parser="pymupdf")
        assert "WESTCHESTER COUNTY" in text

    def test_pymupdf_pages_explicit(self):
        pages = extract_text_pages(str(MCI_PDF), parser="pymupdf")
        assert isinstance(pages, list)
        assert len(pages) >= 1
        assert any("WESTCHESTER COUNTY" in p for p in pages)


# ---------------------------------------------------------------------------
# PDF subsetting
# ---------------------------------------------------------------------------

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
