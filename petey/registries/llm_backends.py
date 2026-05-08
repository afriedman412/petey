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

API_LLM_BACKENDS: dict[str, dict] = {
    # Example:
    # "myhost": {
    #     "client": "openai",          # which builder to use
    #     "base_url": "https://...",    # OpenAI-compatible endpoint
    #     "api_key_env": "MYHOST_KEY", # env var for the key
    # },
}

PLUGIN_LLM_BACKENDS: dict[str, str] = {}
