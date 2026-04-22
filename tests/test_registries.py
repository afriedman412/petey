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

    def test_api_llm_backends_are_live(self):
        """Mutating API_LLM_BACKENDS at runtime should be visible
        to LLM_BACKENDS lookups (no module reload required)."""
        from petey.extract import API_LLM_BACKENDS, LLM_BACKENDS
        assert "ephemeral-test-backend" not in LLM_BACKENDS
        try:
            API_LLM_BACKENDS["ephemeral-test-backend"] = {
                "client": "openai",
                "api_key_env": "NONEXISTENT_KEY",
            }
            assert "ephemeral-test-backend" in LLM_BACKENDS
            assert LLM_BACKENDS.get(
                "ephemeral-test-backend",
            ) is not None
        finally:
            del API_LLM_BACKENDS["ephemeral-test-backend"]
        assert "ephemeral-test-backend" not in LLM_BACKENDS


class TestModelsRegistryConfig:
    """Models with a ``config`` dict should pass it to the builder."""

    def test_config_from_model_entry_reaches_builder(self):
        from petey.extract import MODELS, _make_client
        MODELS["_t_azure"] = {
            "provider": "azure_openai",
            "config": {
                "api_version": "2024-06-01",
                "azure_endpoint": "https://example.azure.test",
                "api_key_env": "T_AZURE_KEY",
            },
        }
        try:
            with patch.dict(os.environ, {"T_AZURE_KEY": "secret"}):
                with patch(
                    "petey.extract.AsyncAzureOpenAI",
                ) as mock_az:
                    with patch(
                        "petey.extract.instructor",
                    ) as mock_inst:
                        mock_inst.from_openai.return_value = "cli"
                        _make_client("_t_azure")
            mock_az.assert_called_once_with(
                api_key="secret",
                api_version="2024-06-01",
                azure_endpoint="https://example.azure.test",
            )
        finally:
            del MODELS["_t_azure"]

    def test_api_model_alias(self):
        """``model`` field in MODELS entry should override the key
        as the identifier sent to the API."""
        from petey.extract import MODELS, _resolve_api_model
        MODELS["_alias_test"] = {
            "provider": "azure_openai",
            "model": "actual-deployment-name",
            "config": {},
        }
        try:
            assert _resolve_api_model(
                "_alias_test",
            ) == "actual-deployment-name"
            assert _resolve_api_model(
                "unregistered-model",
            ) == "unregistered-model"
            assert _resolve_api_model("gpt-4.1-mini") == "gpt-4.1-mini"
        finally:
            del MODELS["_alias_test"]

    def test_two_tenants_same_provider_coexist(self):
        """Same provider (azure_openai), two different deployments,
        no env mutation between calls."""
        from petey.extract import MODELS, _make_client
        MODELS["_t_a"] = {
            "provider": "azure_openai",
            "config": {
                "api_version": "2024-06-01",
                "azure_endpoint": "https://a.azure.test",
                "api_key_env": "T_A_KEY",
            },
        }
        MODELS["_t_b"] = {
            "provider": "azure_openai",
            "config": {
                "api_version": "2024-10-21",
                "azure_endpoint": "https://b.azure.test",
                "api_key_env": "T_B_KEY",
            },
        }
        try:
            env = {"T_A_KEY": "ka", "T_B_KEY": "kb"}
            with patch.dict(os.environ, env):
                with patch(
                    "petey.extract.AsyncAzureOpenAI",
                ) as mock_az:
                    with patch(
                        "petey.extract.instructor",
                    ) as mock_inst:
                        mock_inst.from_openai.return_value = "cli"
                        _make_client("_t_a")
                        _make_client("_t_b")
            calls = mock_az.call_args_list
            assert calls[0].kwargs["api_key"] == "ka"
            assert calls[0].kwargs["azure_endpoint"] == "https://a.azure.test"
            assert calls[1].kwargs["api_key"] == "kb"
            assert calls[1].kwargs["azure_endpoint"] == "https://b.azure.test"
        finally:
            del MODELS["_t_a"]
            del MODELS["_t_b"]


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

    def test_unknown_model_raises(self):
        from petey.extract import _get_provider
        with pytest.raises(ValueError, match="Unknown model"):
            _get_provider("some-other-model")

    def test_unknown_model_with_explicit_backend(self):
        from petey.extract import _get_provider
        assert _get_provider(
            "some-other-model", llm_backend="openai",
        ) == "openai"

    def test_azure_openai_requires_explicit_routing(self):
        from petey.extract import _get_provider
        assert _get_provider(
            "my-azure-deployment", llm_backend="azure_openai",
        ) == "azure_openai"

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
