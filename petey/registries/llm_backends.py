"""LLM backend registries — data only.

Built-in backends (``openai``, ``azure_openai``, ``anthropic``,
``litellm``) are wired up in ``petey.extract``. This module declares
the two pluggable LLM backend registries:

- ``API_LLM_BACKENDS``: configs for OpenAI-compatible HTTP endpoints
  (vLLM, Ollama, Together, custom hosts, …).
- ``PLUGIN_LLM_BACKENDS``: lazy-imported local LLM client factories.

Both are mutable dicts — register at runtime by mutating them.

To add a new provider that speaks the OpenAI protocol::

    from petey.registries.llm_backends import API_LLM_BACKENDS
    API_LLM_BACKENDS["myhost"] = {
        "client": "openai",                # which builder to use
        "base_url": "https://my-host.com/v1",
        "api_key_env": "MYHOST_API_KEY",
    }
"""

import instructor

API_LLM_BACKENDS: dict[str, dict] = {
    # OpenAI-compatible HTTP endpoints. Each speaks the OpenAI
    # protocol well enough that we don't need a dedicated builder —
    # just a base_url, an API-key env var, and (where the provider's
    # tool-call shim is shaky) Mode.JSON to keep the surface stable.
    "deepseek": {
        "client": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "mode": instructor.Mode.JSON,
    },
    "mistral": {
        "client": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "mode": instructor.Mode.JSON,
    },
    "together": {
        "client": "openai",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "mode": instructor.Mode.JSON,
    },
    "openrouter": {
        "client": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "mode": instructor.Mode.JSON,
    },
    "fireworks": {
        "client": "openai",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "mode": instructor.Mode.JSON,
    },
    "groq": {
        "client": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "mode": instructor.Mode.JSON,
    },
}

PLUGIN_LLM_BACKENDS: dict[str, str] = {}
