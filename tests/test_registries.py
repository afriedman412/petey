"""Tests for parser/LLM registries, plugins, backend options, and provider detection."""
import asyncio
import os
from pathlib import Path
from unittest.mock import (
    MagicMock, patch,
)

import pytest

from petey import extract_text_pages

FIXTURES = Path(__file__).parent / "fixtures"
MCI_PDF = FIXTURES / "mci_page1.pdf"


class TestRegistries:
    """Verify all expected backends appear in the registries."""

    def test_parser_registry_has_local_backends(self):
        from petey.extract import PARSERS
        for name in ["pymupdf", "pdfplumber"]:
            assert name in PARSERS, f"Missing parser: {name}"

    def test_parser_registry_has_api_backends(self):
        from petey.extract import PARSERS
        assert "datalab" in PARSERS

    def test_parser_registry_has_plugin_backends(self):
        from petey.extract import PARSERS
        assert "docling" in PARSERS

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

    def test_api_parsers_are_callable(self):
        from petey.extract import PARSERS
        for name in ["datalab"]:
            assert callable(PARSERS[name])

    def test_local_parsers_are_sync(self):
        from petey.extract import PARSERS
        for name in ["pymupdf", "pdfplumber"]:
            assert not asyncio.iscoroutinefunction(
                PARSERS[name]
            ), f"Parser '{name}' should be sync"

    def test_api_parsers_are_async(self):
        from petey.extract import PARSERS, API_PARSERS
        for name in API_PARSERS:
            assert asyncio.iscoroutinefunction(
                PARSERS[name]
            ), f"Parser '{name}' should be async"

    def test_datalab_uses_correct_endpoint(self):
        from petey.extract import API_PARSERS
        assert "datalab" in API_PARSERS
        assert "datalab.to" in API_PARSERS["datalab"]["endpoint"]

    def test_datalab_has_marker_alias(self):
        """Backwards-compat: 'marker' should still resolve to datalab."""
        from petey.extract import PARSERS
        assert "marker" in PARSERS
        assert PARSERS["marker"] is PARSERS["datalab"]


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


class TestLitellmRouting:
    def test_fireworks_routes_to_litellm(self):
        from petey.extract import _get_provider
        assert _get_provider("fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct") == "litellm"

    def test_deepseek_routes_to_litellm(self):
        from petey.extract import _get_provider
        assert _get_provider("deepseek/deepseek-chat") == "litellm"


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


class TestModelKwargs:
    def test_gpt4_returns_max_tokens_and_temperature(self):
        from petey.extract import _model_kwargs
        kw = _model_kwargs("gpt-4.1-mini")
        assert kw == {"max_tokens": 4096, "temperature": 0}

    def test_gpt4_custom_max_tokens(self):
        from petey.extract import _model_kwargs
        kw = _model_kwargs("gpt-4.1", max_tokens=8192)
        assert kw == {"max_tokens": 8192, "temperature": 0}

    def test_gpt5_returns_max_completion_tokens(self):
        from petey.extract import _model_kwargs
        kw = _model_kwargs("gpt-5-mini")
        assert kw == {"max_completion_tokens": 4096}
        assert "temperature" not in kw

    def test_gpt5_full(self):
        from petey.extract import _model_kwargs
        kw = _model_kwargs("gpt-5")
        assert "max_completion_tokens" in kw
        assert "max_tokens" not in kw

    def test_gpt54(self):
        from petey.extract import _model_kwargs
        kw = _model_kwargs("gpt-5.4-mini")
        assert kw == {"max_completion_tokens": 4096}

    def test_non_gpt_model(self):
        from petey.extract import _model_kwargs
        kw = _model_kwargs("claude-sonnet-4-6")
        assert kw == {"max_tokens": 4096, "temperature": 0}


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
