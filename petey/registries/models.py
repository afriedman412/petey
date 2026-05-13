"""Model registry — data, plus a user-config loader.

Built-in entries below are augmented at import time by an optional
user-side YAML file (``~/.petey/models.yaml`` by default, or
``$PETEY_MODELS``). User-config entries override built-ins on key
collision. Set ``PETEY_DISABLE_USER_MODELS=1`` to skip the file
(used in tests to keep the registry deterministic).

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


# --- User-config loader ---
#
# Tracks where each registry entry came from so `petey models list`
# can show built-in vs. user-config provenance.
import os
from pathlib import Path

_BUILTIN_KEYS = frozenset(MODELS)
_MODEL_SOURCES: dict[str, str] = {k: "built-in" for k in MODELS}

DEFAULT_USER_MODELS_PATH = Path.home() / ".petey" / "models.yaml"


def user_models_path() -> Path:
    """Resolve the user-config models YAML path.

    ``PETEY_MODELS`` env var wins; otherwise defaults to
    ``~/.petey/models.yaml``.
    """
    env = os.environ.get("PETEY_MODELS")
    return Path(env) if env else DEFAULT_USER_MODELS_PATH


def load_models_file(path) -> dict[str, dict]:
    """Parse a models YAML file and return the dict.

    Raises ``ValueError`` if the file isn't a mapping of
    ``name -> entry``. Lazy-imports yaml to keep this module
    cheap to import.
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Models config at {path} must be a YAML mapping "
            f"(name -> {{provider: ..., ...}}); got {type(data).__name__}."
        )
    for name, entry in data.items():
        if not isinstance(entry, dict) or "provider" not in entry:
            raise ValueError(
                f"Models config at {path}: entry '{name}' must be a "
                f"dict with at least a 'provider' field."
            )
        if "default" in entry and not isinstance(entry["default"], bool):
            raise ValueError(
                f"Models config at {path}: entry '{name}' has "
                f"'default: {entry['default']!r}' — must be true or false."
            )
    return data


_DEFAULT_MODEL: str | None = None


def register_models(entries: dict[str, dict], source: str) -> list[str]:
    """Merge ``entries`` into MODELS and tag each with ``source``.

    If any entry sets ``default: true``, mark it as the registry-wide
    default. When multiple entries in one call claim the default, warn
    and keep the first one.

    Returns the list of names that were registered (or overridden).
    """
    global _DEFAULT_MODEL
    registered = []
    local_default: str | None = None
    for name, entry in entries.items():
        entry = dict(entry)
        is_default = entry.pop("default", False)
        MODELS[name] = entry
        _MODEL_SOURCES[name] = source
        registered.append(name)
        if is_default:
            if local_default is None:
                local_default = name
            else:
                import warnings
                warnings.warn(
                    f"Models config at {source}: multiple entries set "
                    f"'default: true' ({local_default!r} and {name!r}). "
                    f"Keeping {local_default!r} as the default.",
                    stacklevel=2,
                )
    if local_default is not None:
        _DEFAULT_MODEL = local_default
    return registered


def default_model() -> str | None:
    """Return the model name flagged ``default: true`` in user config,
    or ``None`` if no user-config entry has claimed the default."""
    return _DEFAULT_MODEL


def model_source(name: str) -> str:
    """Return the source label for a registered model
    ('built-in', a path, or 'unknown')."""
    return _MODEL_SOURCES.get(name, "unknown")


def _autoload_user_models() -> None:
    """Import-time hook — load the user-config file if present.

    Silently no-ops if the file doesn't exist or
    ``PETEY_DISABLE_USER_MODELS`` is set. Bad YAML raises so users
    notice broken config rather than silently losing entries.
    """
    if os.environ.get("PETEY_DISABLE_USER_MODELS"):
        return
    path = user_models_path()
    if not path.exists():
        return
    entries = load_models_file(path)
    register_models(entries, source=str(path))


_autoload_user_models()
