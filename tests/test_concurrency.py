"""Tests for ConcurrencyManager and async extraction concurrency."""
import asyncio
import os
from pathlib import Path
from unittest.mock import (
    AsyncMock, MagicMock, patch,
)

import pytest

from petey import extract_text, extract_text_pages

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


def _add(a, b):
    return a + b


def _triple(x):
    return x * 3


def _one():
    return 1


def _mock_llm_client():
    """Build a mock instructor client for LLM calls."""
    mock_result = MagicMock()
    mock_result.model_dump.return_value = {"field": "value"}
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=mock_result,
    )
    return client


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

        async def fake_parse(path, parser):
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
    def test_datalab_parser_routes_to_api_pool(self, tmp_path):
        """parser='datalab' (async registry entry) should
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
                    return_value="datalab text",
                ):
                    with patch(
                        "petey.extract._make_client",
                        return_value=_mock_llm_client(),
                    ):
                        asyncio.run(
                            extract_async(
                                str(pdf), M,
                                parser="datalab",
                            )
                        )

        assert calls["run"] >= 1, (
            "datalab (async) should route through run()"
        )
        assert calls["run_cpu"] == 0, (
            "datalab should NOT use run_cpu"
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


class TestApiIntegration:
    """Test that API backends integrate with extract_text."""

    @patch.dict(os.environ, {"DATALAB_API_KEY": "test-key"})
    def test_datalab_parser_via_extract_text(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        with patch(
            "petey.extract._api_post",
            new_callable=AsyncMock,
            return_value="datalab output",
        ):
            text = extract_text(
                str(pdf), parser="datalab",
            )
        assert text == "datalab output"

    def test_unknown_parser_raises(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake pdf")
        with pytest.raises(ValueError, match="not found"):
            extract_text(str(pdf), parser="buttparse")


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
        """parser='datalab' should go through mgr.run()
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
                    return_value="datalab text",
                ):
                    with patch(
                        "petey.extract._make_client",
                        return_value=_mock_llm_client(),
                    ):
                        asyncio.run(
                            extract_pages_async(
                                str(MCI_PDF), M,
                                parser="datalab",
                            )
                        )

        assert calls["run"] >= 1, (
            "datalab should route through run()"
        )
        assert calls["run_cpu"] == 0, (
            "datalab should NOT use run_cpu"
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
