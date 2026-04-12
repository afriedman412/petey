"""Tests for remote API backend infrastructure: auth, posting, polling, parsing."""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import (
    AsyncMock, MagicMock, patch,
)

import pytest


FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


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
