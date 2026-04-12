"""Tests for schema inference (text and vision paths)."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from petey import extract_text_pages

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _OpenAILike:
    """Minimal spec for an OpenAI-style raw client (no .messages)."""
    class _Chat:
        class _Completions:
            async def create(self, **kw): ...
        completions = _Completions()
    chat = _Chat()


def _mock_openai_raw_client(response_json: str):
    """Build a mock that looks like an OpenAI raw client."""
    msg = MagicMock()
    msg.content = response_json
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    raw = MagicMock(spec=_OpenAILike)
    raw.chat.completions.create = AsyncMock(return_value=resp)
    client = MagicMock()
    client.client = raw
    return client, raw


def _mock_anthropic_raw_client(response_json: str):
    """Build a mock that looks like an Anthropic raw client."""
    block = MagicMock()
    block.text = response_json
    resp = MagicMock()
    resp.content = [block]
    raw = MagicMock()
    raw.messages = MagicMock()
    raw.messages.create = AsyncMock(return_value=resp)
    client = MagicMock()
    client.client = raw
    return client, raw


DUMMY_SCHEMA_JSON = '{"name": "test", "mode": "table", "fields": {"f": {"type": "string"}}}'


# ---------------------------------------------------------------------------
# infer_schema_async — page selection
# ---------------------------------------------------------------------------

class TestInferSchemaPageSelection:
    """Test that infer_schema_async sends the right pages to the LLM."""

    def _run(self, **kwargs):
        from petey.extract import infer_schema_async
        client, raw = _mock_openai_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            result = asyncio.run(
                infer_schema_async(str(MCI_PDF), api_key="k", **kwargs)
            )
        # Return the user message sent to the LLM
        call_kwargs = raw.chat.completions.create.call_args
        return call_kwargs.kwargs["messages"][1]["content"]

    def test_no_page_range_sends_first_pages(self):
        """Without page_range, sends first max_pages pages."""
        pages = extract_text_pages(str(MCI_PDF))
        msg = self._run(max_pages=1)
        # Should contain page 1 content
        assert pages[0][:50] in msg

    def test_page_range_selects_pages(self):
        """page_range='1' should only send page 1."""
        pages = extract_text_pages(str(MCI_PDF))
        msg = self._run(page_range="1", max_pages=2)
        assert pages[0][:50] in msg

    def test_header_pages_prepended_with_separator(self):
        """header_pages=1 should prepend header with HEADER END marker."""
        msg = self._run(header_pages=1, max_pages=1)
        assert "---HEADER END---" in msg

    def test_no_header_pages_no_separator(self):
        """Without header_pages, no HEADER END marker."""
        msg = self._run(header_pages=0, max_pages=2)
        assert "---HEADER END---" not in msg

    def test_page_range_with_header_pages(self):
        """page_range='1-2' with header_pages=1: page 1 is header,
        page 2 is content."""
        pages = extract_text_pages(str(MCI_PDF))
        if len(pages) < 2:
            pytest.skip("test PDF has only 1 page")
        msg = self._run(page_range="1-2", header_pages=1, max_pages=1)
        assert "---HEADER END---" in msg
        # Header is page 1 content, content is page 2
        assert pages[0][:50] in msg
        assert pages[1][:50] in msg


class TestInferSchemaClientRouting:
    """Test that infer_schema_async uses the right API for each provider."""

    def test_openai_client_uses_chat_completions(self):
        from petey.extract import infer_schema_async
        client, raw = _mock_openai_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_async(str(MCI_PDF), api_key="k")
            )
        raw.chat.completions.create.assert_called_once()

    def test_anthropic_client_uses_messages(self):
        from petey.extract import infer_schema_async
        client, raw = _mock_anthropic_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_async(str(MCI_PDF), api_key="k")
            )
        raw.messages.create.assert_called_once()

    def test_anthropic_passes_system_separately(self):
        """Anthropic client should get system as a kwarg, not in messages."""
        from petey.extract import infer_schema_async, INFER_SCHEMA_SYSTEM
        client, raw = _mock_anthropic_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_async(str(MCI_PDF), api_key="k")
            )
        call_kwargs = raw.messages.create.call_args.kwargs
        assert call_kwargs["system"] == INFER_SCHEMA_SYSTEM
        # Messages should only have user role, no system
        for m in call_kwargs["messages"]:
            assert m["role"] != "system"


# ---------------------------------------------------------------------------
# infer_schema_vision_async
# ---------------------------------------------------------------------------

class TestInferSchemaVision:
    """Test vision-based schema inference."""

    def test_openai_sends_image_url(self):
        """OpenAI path should send image_url content parts."""
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_openai_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_vision_async(str(MCI_PDF), api_key="k")
            )
        call_kwargs = raw.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        # Should have system + user messages
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # User message should be a list of content parts
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        has_image = any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in user_content
        )
        assert has_image, "Should contain image_url parts"

    def test_anthropic_sends_image_source(self):
        """Anthropic path should send source-type image parts."""
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_anthropic_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_vision_async(str(MCI_PDF), api_key="k")
            )
        call_kwargs = raw.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        user_content = messages[0]["content"]
        assert isinstance(user_content, list)
        has_image = any(
            isinstance(part, dict) and part.get("type") == "image"
            for part in user_content
        )
        assert has_image, "Should contain image source parts"

    def test_vision_max_pages(self):
        """max_pages should limit the number of images sent."""
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_openai_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_vision_async(
                    str(MCI_PDF), api_key="k", max_pages=1,
                )
            )
        call_kwargs = raw.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        image_parts = [
            p for p in user_content
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        assert len(image_parts) <= 1

    def test_vision_page_range(self):
        """page_range should select specific pages."""
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_openai_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_vision_async(
                    str(MCI_PDF), api_key="k",
                    page_range="1", max_pages=5,
                )
            )
        call_kwargs = raw.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][1]["content"]
        image_parts = [
            p for p in user_content
            if isinstance(p, dict) and p.get("type") == "image_url"
        ]
        # MCI_PDF has 1 page, page_range="1" → 1 image
        assert len(image_parts) == 1

    def test_vision_anthropic_page_range(self):
        """Anthropic vision with page_range."""
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_anthropic_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_vision_async(
                    str(MCI_PDF), api_key="k",
                    page_range="1", max_pages=5,
                )
            )
        call_kwargs = raw.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        image_parts = [
            p for p in user_content
            if isinstance(p, dict) and p.get("type") == "image"
        ]
        assert len(image_parts) == 1

    def test_vision_header_pages(self):
        """header_pages should be prepended as images."""
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_openai_raw_client(DUMMY_SCHEMA_JSON)
        with patch("petey.extract._make_client", return_value=client):
            asyncio.run(
                infer_schema_vision_async(
                    str(MCI_PDF), api_key="k",
                    header_pages=1, max_pages=1,
                )
            )
        # Should succeed (not crash) even with header_pages
        raw.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

class TestInferSchemaJsonParsing:
    """Test that infer_schema handles various LLM response formats."""

    def _run_with_response(self, response_text):
        from petey.extract import infer_schema_async
        client, raw = _mock_openai_raw_client(response_text)
        with patch("petey.extract._make_client", return_value=client):
            return asyncio.run(
                infer_schema_async(str(MCI_PDF), api_key="k")
            )

    def test_plain_json(self):
        result = self._run_with_response(DUMMY_SCHEMA_JSON)
        assert result["name"] == "test"

    def test_json_in_code_block(self):
        response = '```json\n' + DUMMY_SCHEMA_JSON + '\n```'
        result = self._run_with_response(response)
        assert result["name"] == "test"

    def test_prose_before_code_block(self):
        """4o-mini style: prose text then a code block."""
        response = (
            "Based on the document, here is a suggested schema:\n\n"
            "```json\n" + DUMMY_SCHEMA_JSON + "\n```"
        )
        result = self._run_with_response(response)
        assert result["name"] == "test"

    def test_prose_before_and_after_code_block(self):
        response = (
            "Here's the schema:\n\n"
            "```json\n" + DUMMY_SCHEMA_JSON + "\n```\n\n"
            "Let me know if you'd like changes."
        )
        result = self._run_with_response(response)
        assert result["name"] == "test"

    def test_code_block_without_json_tag(self):
        response = "```\n" + DUMMY_SCHEMA_JSON + "\n```"
        result = self._run_with_response(response)
        assert result["name"] == "test"

    def test_empty_response_raises(self):
        from petey.extract import infer_schema_async
        client, raw = _mock_openai_raw_client("")
        # Make the response content empty
        raw.chat.completions.create.return_value.choices[
            0
        ].message.content = ""
        with patch(
            "petey.extract._make_client", return_value=client,
        ):
            with pytest.raises(ValueError, match="empty response"):
                asyncio.run(
                    infer_schema_async(str(MCI_PDF), api_key="k")
                )

    def test_none_response_raises(self):
        from petey.extract import infer_schema_async
        client, raw = _mock_openai_raw_client("")
        raw.chat.completions.create.return_value.choices[
            0
        ].message.content = None
        with patch(
            "petey.extract._make_client", return_value=client,
        ):
            with pytest.raises(ValueError, match="empty response"):
                asyncio.run(
                    infer_schema_async(str(MCI_PDF), api_key="k")
                )

    def test_invalid_json_raises_with_preview(self):
        from petey.extract import infer_schema_async
        client, raw = _mock_openai_raw_client(
            "This is not JSON at all"
        )
        with patch(
            "petey.extract._make_client", return_value=client,
        ):
            with pytest.raises(ValueError, match="invalid JSON"):
                asyncio.run(
                    infer_schema_async(str(MCI_PDF), api_key="k")
                )


class TestInferSchemaVisionJsonParsing:
    """JSON parsing tests for the vision path (mirrors TestInferSchemaJsonParsing)."""

    def _run_with_response(self, response_text):
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_openai_raw_client(response_text)
        with patch(
            "petey.extract._make_client", return_value=client,
        ):
            return asyncio.run(
                infer_schema_vision_async(
                    str(MCI_PDF), api_key="k",
                )
            )

    def test_plain_json(self):
        result = self._run_with_response(DUMMY_SCHEMA_JSON)
        assert result["name"] == "test"

    def test_json_in_code_block(self):
        response = '```json\n' + DUMMY_SCHEMA_JSON + '\n```'
        result = self._run_with_response(response)
        assert result["name"] == "test"

    def test_prose_before_code_block(self):
        response = (
            "Based on the document, here is a suggested schema:\n\n"
            "```json\n" + DUMMY_SCHEMA_JSON + "\n```"
        )
        result = self._run_with_response(response)
        assert result["name"] == "test"

    def test_code_block_without_json_tag(self):
        response = "```\n" + DUMMY_SCHEMA_JSON + "\n```"
        result = self._run_with_response(response)
        assert result["name"] == "test"

    def test_empty_response_raises(self):
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_openai_raw_client("")
        raw.chat.completions.create.return_value.choices[
            0
        ].message.content = ""
        with patch(
            "petey.extract._make_client", return_value=client,
        ):
            with pytest.raises(ValueError, match="empty response"):
                asyncio.run(
                    infer_schema_vision_async(
                        str(MCI_PDF), api_key="k",
                    )
                )

    def test_invalid_json_raises_with_preview(self):
        from petey.extract import infer_schema_vision_async
        client, raw = _mock_openai_raw_client(
            "This is not JSON at all"
        )
        with patch(
            "petey.extract._make_client", return_value=client,
        ):
            with pytest.raises(ValueError, match="invalid JSON"):
                asyncio.run(
                    infer_schema_vision_async(
                        str(MCI_PDF), api_key="k",
                    )
                )
