"""Model registry — data only.

Explicit registration of known models. Each entry in ``MODELS`` is a
dict with:

  provider (required)
      Name of the LLM backend. Must be a key in ``LLM_BACKENDS``.
      Built-ins: "openai", "azure_openai", "anthropic", "litellm".

  config (optional)
      Dict of kwargs passed to the backend *builder* (not the
      completion call). Lets each model carry its own endpoint,
      API version, organisation, and API-key env var — so multiple
      deployments of the same backend can coexist in one process
      without mutating os.environ.

  kwargs (optional)
      Dict of kwargs passed to every ``chat.completions.create()``
      call. Overrides the default ``{"max_tokens": 4096,
      "temperature": 0}``. Use this for reasoning models that need
      ``max_completion_tokens`` and reject ``temperature``.

  model (optional)
      The identifier the API expects, if it differs from the
      registry key. Lets a registry key serve as an alias
      (e.g. ``tenant-a-gpt-4o`` → Azure deployment ``gpt-4o``).

Example — one process, two Azure tenants::

    MODELS["tenant-a-gpt-4o"] = {
        "provider": "azure_openai",
        "config": {
            "api_version": "2024-06-01",
            "azure_endpoint": "https://tenant-a.openai.azure.com",
            "api_key_env": "TENANT_A_API_KEY",
        },
    }
    MODELS["tenant-b-gpt-5"] = {
        "provider": "azure_openai",
        "config": {
            "api_version": "2024-10-21",
            "azure_endpoint": "https://tenant-b.openai.azure.com",
            "api_key_env": "TENANT_B_API_KEY",
        },
        "kwargs": {"max_completion_tokens": 4096},
    }
"""

_DEFAULT_MODEL_KWARGS = {"max_tokens": 4096, "temperature": 0}
_REASONING_MODEL_KWARGS = {"max_completion_tokens": 4096}

MODELS: dict[str, dict] = {
    # OpenAI
    "gpt-4.1":      {"provider": "openai"},
    "gpt-4.1-mini": {"provider": "openai"},
    "gpt-4o":       {"provider": "openai"},
    "gpt-4o-mini":  {"provider": "openai"},
    "gpt-5":        {"provider": "openai", "kwargs": _REASONING_MODEL_KWARGS},
    "gpt-5-mini":   {"provider": "openai", "kwargs": _REASONING_MODEL_KWARGS},
    # Anthropic
    "claude-opus-4-7":           {"provider": "anthropic"},
    "claude-sonnet-4-6":         {"provider": "anthropic"},
    "claude-haiku-4-5-20251001": {"provider": "anthropic"},
    # Local — Ollama (deck-defining benchmark path)
    "qwen3-4b":          {"provider": "ollama", "model": "qwen3:4b"},
    "qwen2.5-3b":        {"provider": "ollama", "model": "qwen2.5:3b"},
    "ollama/qwen3:4b":   {"provider": "ollama", "model": "qwen3:4b"},
    "ollama/qwen2.5:3b": {"provider": "ollama", "model": "qwen2.5:3b"},
    # Gemini direct
    "gemini-2.5-flash":        {"provider": "gemini",
                                "model": "gemini-2.5-flash"},
    "gemini/gemini-2.5-flash": {"provider": "gemini",
                                "model": "gemini-2.5-flash"},
    # OpenAI-compat (config-only, via API_LLM_BACKENDS)
    "deepseek/deepseek-chat":     {"provider": "deepseek",
                                   "model": "deepseek-chat"},
    "deepseek/deepseek-reasoner": {"provider": "deepseek",
                                   "model": "deepseek-reasoner"},
}
