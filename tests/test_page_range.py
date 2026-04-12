"""Tests for page range parsing and header page fallback."""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


class TestParsePageRange:
    def test_single_page(self):
        from petey.extract import _parse_page_range
        assert _parse_page_range("3", 10) == [2]

    def test_range(self):
        from petey.extract import _parse_page_range
        assert _parse_page_range("2-5", 10) == [1, 2, 3, 4]

    def test_comma_list(self):
        from petey.extract import _parse_page_range
        assert _parse_page_range("1,3,5", 10) == [0, 2, 4]

    def test_mixed(self):
        from petey.extract import _parse_page_range
        assert _parse_page_range("1,3-5,8", 10) == [0, 2, 3, 4, 7]

    def test_clamps_to_total(self):
        from petey.extract import _parse_page_range
        result = _parse_page_range("1-100", 5)
        assert result == [0, 1, 2, 3, 4]


def _mock_llm_client():
    """Build a mock instructor client for LLM calls."""
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {"field": "value"}
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=mock_result,
    )
    return client


class TestExtractPagesHeaderFallback:
    """When header_pages >= content pages, headers become content."""

    def test_single_page_with_header_pages_1(self):
        """1-page PDF with header_pages=1: page should be extracted
        as content (not skipped)."""
        from petey.extract import extract_pages_async
        from pydantic import BaseModel

        class M(BaseModel):
            field: str | None = None

        with patch(
            "petey.extract._make_client",
            return_value=_mock_llm_client(),
        ):
            results = asyncio.run(
                extract_pages_async(
                    str(MCI_PDF), M,
                    header_pages=1,
                    parser="pymupdf",
                )
            )
        # Should have at least 1 result (not empty)
        assert len(results) >= 1

    def test_two_page_with_header_pages_1(self):
        """2-page PDF with header_pages=1 where content <= headers:
        both pages should be content chunks."""
        from petey.extract import extract_pages_async
        from pydantic import BaseModel
        import tempfile
        import fitz

        # Create a 2-page test PDF
        doc = fitz.open()
        for i in range(2):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1} content")
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()

        class M(BaseModel):
            field: str | None = None

        # Return unique data per call to avoid dedup
        call_count = {"n": 0}
        def _unique_mock_client():
            mock_result = MagicMock()
            def _side_effect(**kw):
                call_count["n"] += 1
                r = MagicMock()
                r.model_dump.return_value = {
                    "field": f"value_{call_count['n']}",
                }
                return r
            client = MagicMock()
            client.chat.completions.create = AsyncMock(
                side_effect=_side_effect,
            )
            return client

        try:
            with patch(
                "petey.extract._make_client",
                return_value=_unique_mock_client(),
            ):
                results = asyncio.run(
                    extract_pages_async(
                        tmp.name, M,
                        header_pages=1,
                        parser="pymupdf",
                    )
                )
            # Fallback: both pages become content → 2 chunks
            assert len(results) == 2
        finally:
            os.unlink(tmp.name)

    def test_many_pages_no_fallback(self):
        """10-page PDF with header_pages=1: no fallback, 9 content
        chunks."""
        from petey.extract import extract_pages_async
        from pydantic import BaseModel
        import tempfile
        import fitz

        doc = fitz.open()
        for i in range(10):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i+1} content")
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()

        class M(BaseModel):
            field: str | None = None

        # Return unique data per call to avoid dedup
        call_count = {"n": 0}
        def _unique_mock_client():
            def _side_effect(**kw):
                call_count["n"] += 1
                r = MagicMock()
                r.model_dump.return_value = {
                    "field": f"value_{call_count['n']}",
                }
                return r
            client = MagicMock()
            client.chat.completions.create = AsyncMock(
                side_effect=_side_effect,
            )
            return client

        try:
            with patch(
                "petey.extract._make_client",
                return_value=_unique_mock_client(),
            ):
                results = asyncio.run(
                    extract_pages_async(
                        tmp.name, M,
                        header_pages=1,
                        parser="pymupdf",
                    )
                )
            # 9 content pages, no fallback
            assert len(results) == 9
        finally:
            os.unlink(tmp.name)
