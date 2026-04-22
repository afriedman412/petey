"""
Core PDF extraction. No web dependencies.
Configure via environment variables or pass explicitly.

Pipeline architecture::

    PDF
     └─ Parser       (pymupdf | pdfplumber | datalab | unstructured | …)
          └─ LLM     (openai | anthropic | litellm)
               └─ Output  (csv | json | jsonl)

Each layer is swappable via parameters on the public functions.
"""
import asyncio
import base64
from petey.concurrency import get_manager
from functools import partial
import importlib
import json
import os
import tempfile
import warnings

import httpx

import fitz
import pymupdf4llm
import instructor
from pydantic import BaseModel
from openai import AsyncAzureOpenAI, AsyncOpenAI
from anthropic import AsyncAnthropic

SYSTEM = (
    "You extract structured data from documents. "
    "Use null for missing or unreadable values."
)

TEXT_WARN_THRESHOLD = 50_000
SHORT_TEXT_THRESHOLD = 200  # warn before LLM if text is this short


def _check_extraction_quality(
    data: dict,
    text: str,
    label: str = "",
) -> list[str]:
    """Check extraction results for quality issues.

    Returns a list of warning strings (empty if everything looks fine).
    """
    msgs = []
    prefix = f"[petey] {label}: " if label else "[petey] "

    if len(text.strip()) < SHORT_TEXT_THRESHOLD:
        msgs.append(
            f"{prefix}extracted text was very short "
            f"({len(text.strip())} chars) — try a different parser"
        )

    fields = {
        k: v for k, v in data.items()
        if not k.startswith("_")
    }
    if fields:
        null_count = sum(
            1 for v in fields.values() if v is None
        )
        if null_count >= len(fields) * 0.8:
            msgs.append(
                f"{prefix}{null_count}/{len(fields)} fields "
                f"are null"
            )
    return msgs


# --- PDF text extraction backends ---


def _extract_text_pages_pymupdf(pdf_path: str) -> list[str]:
    """Extract per-page text using pymupdf4llm markdown extraction.

    Produces structured markdown output with headers and table formatting,
    optimised for LLM consumption.
    """
    try:
        chunks = pymupdf4llm.to_markdown(
            pdf_path, page_chunks=True, force_text=False,
        )
        return [chunk["text"] for chunk in chunks]
    except Exception:
        # Ghostscript or other pymupdf4llm failure — fall back to plain text
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return pages


def _extract_text_pages_pdfplumber(pdf_path: str) -> list[str]:
    """Extract per-page text using pdfplumber's layout-preserving mode.

    Uses layout=True which positions text spatially, preserving column
    alignment for borderless tables without needing explicit table detection.
    Falls back to plain extract_text() for pages where layout extraction
    returns nothing.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for parser='pdfplumber'. "
            "Install it with: pip install pdfplumber"
        )
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = (
                page.extract_text(layout=True)
                or page.extract_text()
                or ""
            )
            pages.append(text)
    return pages


# --- Remote API backend infrastructure ---
#
# Any HTTP service that accepts a file and returns text can be wired in
# as a parser or OCR backend by adding config to API_PARSERS / API_OCR_BACKENDS.
# No new functions needed — just a dict.
#
# Config keys:
#   endpoint       — URL to POST the file to (required)
#   api_key_env    — env var name for the API key (required)
#   auth_header    — HTTP header name (default: "X-API-Key")
#   auth_prefix    — prepended to key value, e.g. "Bearer" (default: "")
#   request_format — how to send the file (default: "multipart")
#       "multipart"  — standard multipart/form-data file upload
#       "json_b64"   — base64-encoded file in a JSON body
#   file_field     — field name for the file in the request (default: "file")
#   params         — extra form data / JSON fields to include (default: {})
#   response_key   — dot-path into JSON response for the text (default: "markdown")
#   poll           — whether to poll a check URL for async results (default: True)
#   poll_status_key — key to check for completion (default: "status")
#   poll_done_value — value that means done (default: "complete")
#   poll_check_key  — key containing the poll URL (default: "request_check_url")
#   poll_url_template — build poll URL from check_key value via str.format()
#                       e.g. "https://api.example.com/job/{id}/result"
#                       when set, poll_check_key is the value to interpolate
#   poll_header_key — read the poll URL from a response header instead
#                     of the JSON body (e.g. "Operation-Location" for Azure)
#   endpoint_env   — env var containing the base URL (for per-user endpoints)
#   endpoint_suffix — path appended to endpoint_env value
#   timeout        — max seconds to wait for poll (default: 240)
#
# request_format options:
#   "multipart"  — standard multipart/form-data file upload (default)
#   "json_b64"   — base64-encoded file in a JSON body
#   "raw"        — raw file bytes with Content-Type header
#
# response_key patterns:
#   "markdown"    — simple top-level key
#   "result.text" — dot-separated nested key
#   "[].text"     — join text from each element in an array response

API_PARSERS: dict[str, dict] = {
    "datalab": {
        "name": "Datalab",
        "role": "parser",
        "endpoint": "https://www.datalab.to/api/v1/convert",
        "api_key_env": "DATALAB_API_KEY",
        "auth_header": "X-API-Key",
        "params": {"output_format": "markdown"},
        "response_key": "markdown",
        "poll": True,
    },
    "unstructured_api": {
        "name": "Unstructured API",
        "role": "parser",
        "endpoint": "https://api.unstructuredapp.io/general/v0/general",
        "api_key_env": "UNSTRUCTURED_API_KEY",
        "auth_header": "unstructured-api-key",
        "params": {"strategy": "auto"},
        "response_key": "[].text",
        "poll": False,
    },
    "azure_documentai": {
        "name": "Azure Document Intelligence",
        "role": "parser",
        "endpoint_env": "AZURE_DOCUMENT_ENDPOINT",
        "endpoint_suffix": (
            "/documentintelligence"
            "/documentModels/prebuilt-read:analyze"
            "?api-version=2024-11-30"
        ),
        "api_key_env": "AZURE_DOCUMENT_KEY",
        "auth_header": "Ocp-Apim-Subscription-Key",
        "request_format": "raw",
        "response_key": "analyzeResult.content",
        "poll": True,
        "poll_header_key": "Operation-Location",
        "poll_status_key": "status",
        "poll_done_value": "succeeded",
        "timeout": 120,
    },
}


def _resolve_response(data, key_path: str) -> str:
    """Extract text from an API response.

    ``"markdown"`` → ``data["markdown"]``
    ``"result.text"`` → ``data["result"]["text"]``
    ``"[].text"``  → join ``item["text"]`` from a list of dicts
    """
    # Handle array responses: [].key joins text from each element
    if key_path.startswith("[]."):
        inner_key = key_path[3:]
        if isinstance(data, list):
            parts = [
                str(item.get(inner_key, ""))
                for item in data if isinstance(item, dict)
            ]
            return "\n\n".join(p for p in parts if p)
        return ""
    obj = data
    for part in key_path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part, "")
        else:
            return ""
    return str(obj) if obj else ""


def _build_auth_header(cfg: dict, api_key: str) -> dict:
    """Build the auth header dict from config."""
    header_name = cfg.get("auth_header", "X-API-Key")
    prefix = cfg.get("auth_prefix", "")
    value = f"{prefix} {api_key}".strip() if prefix else api_key
    return {header_name: value}


async def _api_post(
    cfg: dict,
    file_bytes: bytes,
    filename: str,
    content_type: str,
) -> str:
    """POST a file to a remote API and return the extracted text.

    Handles both multipart upload and JSON+base64 request formats,
    sync and async (poll-based) responses.
    """
    api_key = _api_get_key(cfg)
    headers = _build_auth_header(cfg, api_key)
    if "endpoint_env" in cfg:
        base = os.environ.get(cfg["endpoint_env"], "")
        if not base:
            raise ValueError(
                f"{cfg['endpoint_env']} environment variable "
                f"is required for {cfg.get('name', 'backend')}."
            )
        endpoint = base.rstrip("/") + cfg.get(
            "endpoint_suffix", "",
        )
    else:
        endpoint = cfg["endpoint"]
    params = cfg.get("params", {})
    request_format = cfg.get("request_format", "multipart")
    file_field = cfg.get("file_field", "file")

    async with httpx.AsyncClient(timeout=60) as client:
        if request_format == "json_b64":
            payload = {
                file_field: base64.b64encode(
                    file_bytes
                ).decode(),
                "filename": filename,
                **params,
            }
            headers["Content-Type"] = "application/json"
            resp = await client.post(
                endpoint,
                content=json.dumps(payload),
                headers=headers,
            )
        elif request_format == "raw":
            headers["Content-Type"] = content_type
            resp = await client.post(
                endpoint,
                content=file_bytes,
                headers=headers,
            )
        else:  # multipart (default)
            resp = await client.post(
                endpoint,
                files={
                    file_field: (
                        filename, file_bytes, content_type,
                    ),
                },
                data=params,
                headers=headers,
            )

        resp.raise_for_status()
        result = resp.json() if resp.headers.get(
            "content-type", "",
        ).startswith("application/json") else {}

        response_key = cfg.get("response_key", "markdown")

        if cfg.get("poll", True):
            check_key = cfg.get(
                "poll_check_key", "request_check_url",
            )
            status_key = cfg.get("poll_status_key", "status")
            done_value = cfg.get(
                "poll_done_value", "complete",
            )
            timeout = cfg.get("timeout", 240)
            poll_interval = 2
            # poll_header_key: read the poll URL from a
            # response header instead of the JSON body
            poll_header_key = cfg.get("poll_header_key")
            poll_url_template = cfg.get("poll_url_template")
            if poll_header_key:
                check_url = resp.headers.get(poll_header_key)
            elif poll_url_template:
                check_value = result.get(check_key)
                if not check_value:
                    raise ValueError(
                        f"API at {endpoint} "
                        f"did not return '{check_key}'"
                    )
                check_url = poll_url_template.format(
                    **{check_key: check_value},
                )
            else:
                check_url = result.get(check_key)
            if not check_url:
                raise ValueError(
                    f"API at {endpoint} "
                    f"did not return '{check_key}'"
                )
            poll_headers = _build_auth_header(cfg, api_key)
            for _ in range(timeout // poll_interval):
                await asyncio.sleep(poll_interval)
                r = await client.get(
                    check_url, headers=poll_headers,
                )
                r.raise_for_status()
                result = r.json()
                if result.get(status_key) == done_value:
                    return _resolve_response(
                        result, response_key,
                    )
            raise TimeoutError(
                f"API at {endpoint} "
                f"timed out after {timeout}s"
            )

        return _resolve_response(result, response_key)


def _api_get_key(cfg: dict) -> str:
    """Resolve the API key from the environment for a backend config."""
    env_var = cfg["api_key_env"]
    api_key = os.environ.get(env_var)
    if not api_key:
        name = cfg.get("name", "unknown")
        role = cfg.get("role", "backend")
        raise ValueError(
            f"Missing API key for {name} ({role}). "
            f"Set {env_var} in your .env file or environment."
        )
    return api_key


async def _parse_pdf_via_api(pdf_path: str, cfg: dict) -> list[str]:
    """Parse a full PDF by uploading it to a remote API."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    text = await _api_post(
        cfg, pdf_bytes, "document.pdf", "application/pdf",
    )
    return [text] if text else [""]


# --- Plugin registries ---
#
# Register local backends that live outside extract.py.  Each entry maps a
# name to a "module.path:callable" string.  The callable is lazy-imported
# the first time someone selects that backend, so heavyweight dependencies
# (like docling) are never loaded unless needed.
#
# Callable contracts:
#   PLUGIN_PARSERS:      (pdf_path: str) -> list[str]  (one string per page)
#   PLUGIN_OCR_BACKENDS: (page: fitz.Page) -> str
#   PLUGIN_LLM_BACKENDS: same as _LLM_CLIENT_BUILDERS — a client factory
#
# Users can add their own:
#   from petey.extract import PLUGIN_PARSERS
#   PLUGIN_PARSERS["my_parser"] = "my_package.pdf:extract_pages"

PLUGIN_PARSERS: dict[str, str] = {
    "docling": "petey.plugins.docling:extract_pages",
    "liteparse": "petey.plugins.liteparse:extract_pages",
    "unstructured": "petey.plugins.unstructured:extract_pages",
    "textract": "petey.plugins.textract:extract_pages",
    "google_documentai": "petey.plugins.google_documentai:extract_pages",
}

PLUGIN_LLM_BACKENDS: dict[str, str] = {}


def _load_plugin(import_path: str):
    """Import and return a plugin callable from a 'module:func' string."""
    module_path, func_name = import_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _make_plugin_loader(import_path: str):
    """Return a wrapper that lazy-imports a plugin callable on first use.

    If the underlying function is async, the wrapper is async too,
    so the concurrency manager routes it through the API pool.
    If sync, routes through the CPU pool.
    """
    _fn = None

    def _lazy_sync(*args, **kwargs):
        nonlocal _fn
        if _fn is None:
            _fn = _load_plugin(import_path)
        return _fn(*args, **kwargs)

    async def _lazy_async(*args, **kwargs):
        nonlocal _fn
        if _fn is None:
            _fn = _load_plugin(import_path)
        return await _fn(*args, **kwargs)

    # Peek at the function to decide which wrapper to return.
    # This imports the module at registry-build time, but the
    # heavy dependencies are still behind lazy imports inside
    # the plugin function itself (try/except ImportError).
    fn = _load_plugin(import_path)
    if asyncio.iscoroutinefunction(fn):
        _fn = fn
        return _lazy_async
    _fn = fn
    return _lazy_sync


# --- Parser registry ---
# Each parser is a callable: (pdf_path: str) -> list[str]
# Local parsers are plain functions; API parsers are built from config.
# Plugin parsers are lazy-imported on first use.

PARSERS = {
    "pymupdf": _extract_text_pages_pymupdf,
    "pdfplumber": _extract_text_pages_pdfplumber,
    **{name: _make_plugin_loader(path)
       for name, path in PLUGIN_PARSERS.items()},
    **{name: partial(_parse_pdf_via_api, cfg=cfg)
       for name, cfg in API_PARSERS.items()},
}

# Backward compatibility alias
PARSERS["marker"] = PARSERS["datalab"]


def extract_text(
    pdf_path: str,
    parser: str = "pymupdf",
    parser_options: dict | None = None,
) -> str:
    """Extract all text from a PDF as a single string.

    Args:
        pdf_path: Path to the PDF file.
        parser: Text extraction backend — see ``PARSERS`` registry.
        parser_options: Extra kwargs passed to the parser callable.

    Returns:
        Extracted text as a single string.
    """
    pages = extract_text_pages(
        pdf_path, parser,
        parser_options=parser_options,
    )
    return "\n\n".join(pages)


def extract_text_pages(
    pdf_path: str,
    parser: str = "pymupdf",
    parser_options: dict | None = None,
) -> list[str]:
    """Extract text from each page of a PDF separately.

    Args:
        pdf_path: Path to the PDF file.
        parser: Text extraction backend — see ``PARSERS`` registry.
        parser_options: Extra kwargs passed to the parser callable.

    Returns:
        List of strings, one per page.
    """
    _p_opts = parser_options or {}

    fn = PARSERS.get(parser)
    if fn is None:
        raise ValueError(
            f"Parser '{parser}' not found. "
            f"Available parsers: {', '.join(PARSERS)}"
        )
    if asyncio.iscoroutinefunction(fn):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError(
                f"Parser '{parser}' is async. Use "
                f"extract_async() or "
                f"extract_pages_async() instead."
            )
        pages = asyncio.run(fn(pdf_path, **_p_opts))
    else:
        pages = fn(pdf_path, **_p_opts)

    return pages


# --- LLM client ---


# --- Model registry ---
#
# Explicit registration of known models. Each entry is a dict with:
#
#   provider (required)
#       Name of the LLM backend. Must be a key in ``LLM_BACKENDS``.
#       Built-ins: "openai", "azure_openai", "anthropic", "litellm".
#
#   config (optional)
#       Dict of kwargs passed to the backend *builder* (not the
#       completion call). Lets each model carry its own endpoint,
#       API version, organisation, and API-key env var — so multiple
#       deployments of the same backend can coexist in one process
#       without mutating os.environ.
#
#   kwargs (optional)
#       Dict of kwargs passed to every ``chat.completions.create()``
#       call. Overrides the default ``{"max_tokens": 4096,
#       "temperature": 0}``. Use this for reasoning models that need
#       ``max_completion_tokens`` and reject ``temperature``.
#
# Example — one process, two Azure tenants:
#
#   MODELS["tenant-a-gpt-4o"] = {
#       "provider": "azure_openai",
#       "config": {
#           "api_version": "2024-06-01",
#           "azure_endpoint": "https://tenant-a.openai.azure.com",
#           "api_key_env": "TENANT_A_API_KEY",
#       },
#   }
#   MODELS["tenant-b-gpt-5"] = {
#       "provider": "azure_openai",
#       "config": {
#           "api_version": "2024-10-21",
#           "azure_endpoint": "https://tenant-b.openai.azure.com",
#           "api_key_env": "TENANT_B_API_KEY",
#       },
#       "kwargs": {"max_completion_tokens": 4096},
#   }

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
}


_LITELLM_PREFIXES = (
    "gemini/", "mistral/", "ollama/", "ollama_chat/",
    "bedrock/", "vertex_ai/", "cohere/", "replicate/",
    "huggingface/", "together_ai/", "openrouter/",
    "fireworks_ai/", "deepseek/",
)


def _infer_provider(model: str) -> str | None:
    """Prefix-based provider inference. Returns None if unrecognised."""
    if model.startswith("claude"):
        return "anthropic"
    if any(model.startswith(p) for p in _LITELLM_PREFIXES):
        return "litellm"
    if (
        model.startswith("gpt-")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    ):
        return "openai"
    return None


def _infer_model_kwargs(model: str) -> dict:
    """Prefix-based model-kwargs inference for unregistered models."""
    if (
        model.startswith("gpt-5")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    ):
        return dict(_REASONING_MODEL_KWARGS)
    return dict(_DEFAULT_MODEL_KWARGS)


def _model_kwargs(model: str) -> dict:
    """Return model-specific API kwargs.

    Consults ``MODELS`` registry first; falls back to prefix inference
    for unregistered models.
    """
    entry = MODELS.get(model)
    if entry is not None:
        return dict(entry.get("kwargs", _DEFAULT_MODEL_KWARGS))
    return _infer_model_kwargs(model)


def _resolve_api_model(model: str) -> str:
    """Resolve a registry key to the identifier the API expects.

    Returns the registry entry's ``model`` field if set, otherwise
    returns the key unchanged. This lets a registry entry serve as
    an alias (e.g. ``tenant-a-gpt-4o`` → Azure deployment
    ``gpt-4o``), so multiple backends exposing the same underlying
    model can coexist under distinct petey-side names.
    """
    entry = MODELS.get(model)
    if entry is not None and "model" in entry:
        return entry["model"]
    return model


def _get_provider(model: str, llm_backend: str | None = None) -> str:
    """Determine which LLM provider to use for ``model``.

    Resolution order:
        1. Explicit ``llm_backend`` kwarg (highest priority).
        2. ``MODELS`` registry entry for the model.
        3. Prefix-based inference (emits a warning — register the
           model to silence it).
        4. Error — the model is unrecognised.

    Args:
        model: Model ID string.
        llm_backend: Optional explicit provider override. Must be a
            key in ``LLM_BACKENDS``.

    Returns:
        Provider name.
    """
    if llm_backend is not None:
        return llm_backend
    entry = MODELS.get(model)
    if entry is not None:
        return entry["provider"]
    inferred = _infer_provider(model)
    if inferred is None:
        raise ValueError(
            f"Unknown model '{model}'. Register it in "
            f"petey.extract.MODELS with "
            f"{{'provider': '<name>', ...}}, or pass "
            f"llm_backend='<name>' explicitly. "
            f"Known models: {', '.join(sorted(MODELS))}"
        )
    warnings.warn(
        f"Model '{model}' is not in the MODELS registry; inferred "
        f"provider='{inferred}' from prefix. Register it in "
        f"petey.extract.MODELS to silence this warning.",
        stacklevel=2,
    )
    return inferred


def _make_client_openai(api_key: str | None = None, **kwargs):
    """Build an instructor-wrapped OpenAI async client.

    Also handles any OpenAI-compatible endpoint (vLLM, Ollama,
    Together, etc.) via the ``base_url`` kwarg.
    """
    env_var = kwargs.get("api_key_env", "OPENAI_API_KEY")
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"Missing API key for OpenAI (llm). "
            f"Set {env_var} in your .env file or environment."
        )
    organization = (
        kwargs.get("organization")
        or os.environ.get("OPENAI_ORGANIZATION")
    )
    client_kwargs = {"api_key": key}
    base_url = kwargs.get("base_url")
    if base_url:
        client_kwargs["base_url"] = base_url
    if organization:
        client_kwargs["organization"] = organization
    return instructor.from_openai(AsyncOpenAI(**client_kwargs))


def _make_client_azure_openai(api_key: str | None = None, **kwargs):
    """Build an instructor-wrapped Azure OpenAI async client.

    Required configuration (kwarg or env var):
        api_key        — OPENAI_API_KEY (or ``api_key_env`` override)
        api_version    — API_VERSION / OPENAI_API_VERSION
        azure_endpoint — OPENAI_API_BASE / AZURE_OPENAI_ENDPOINT

    Optional:
        organization   — OPENAI_ORGANIZATION
    """
    env_var = kwargs.get("api_key_env", "OPENAI_API_KEY")
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"Missing API key for Azure OpenAI (llm). "
            f"Set {env_var} in your .env file or environment."
        )
    api_version = (
        kwargs.get("api_version")
        or os.environ.get("API_VERSION")
        or os.environ.get("OPENAI_API_VERSION")
    )
    if not api_version:
        raise ValueError(
            "Azure OpenAI requires api_version. Set API_VERSION "
            "or OPENAI_API_VERSION, or pass api_version to the "
            "backend config."
        )
    azure_endpoint = (
        kwargs.get("azure_endpoint")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    if not azure_endpoint:
        raise ValueError(
            "Azure OpenAI requires an endpoint. Set "
            "OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT, or pass "
            "azure_endpoint to the backend config."
        )
    organization = (
        kwargs.get("organization")
        or os.environ.get("OPENAI_ORGANIZATION")
    )
    client_kwargs = {
        "api_key": key,
        "api_version": api_version,
        "azure_endpoint": azure_endpoint,
    }
    if organization:
        client_kwargs["organization"] = organization
    return instructor.from_openai(
        AsyncAzureOpenAI(**client_kwargs),
    )


def _make_client_anthropic(api_key: str | None = None, **kwargs):
    """Build an instructor-wrapped Anthropic async client."""
    env_var = kwargs.get("api_key_env", "ANTHROPIC_API_KEY")
    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"Missing API key for Anthropic (llm). "
            f"Set {env_var} in your .env file or environment."
        )
    return instructor.from_anthropic(
        AsyncAnthropic(api_key=key), max_tokens=16384,
    )


def _make_client_litellm(api_key: str | None = None, **kwargs):
    """Build an instructor-wrapped litellm client."""
    import litellm
    litellm.drop_params = True
    return instructor.from_litellm(litellm.acompletion)


# --- LLM backend registry ---
#
# Each backend is a callable: (api_key: str | None, **kwargs) -> instructor client
#
# Built-in backends handle OpenAI, Anthropic, and litellm (100+ providers).
# Any OpenAI-compatible API (vLLM, Ollama, Together, etc.) can be used by
# setting llm_backend="openai" and passing base_url / api_key_env via
# API_LLM_BACKENDS config.
#
# To add a new provider that speaks the OpenAI protocol:
#
#   API_LLM_BACKENDS["myhost"] = {
#       "client": "openai",
#       "base_url": "https://my-host.com/v1",
#       "api_key_env": "MYHOST_API_KEY",
#   }

_LLM_CLIENT_BUILDERS = {
    "openai": _make_client_openai,
    "azure_openai": _make_client_azure_openai,
    "anthropic": _make_client_anthropic,
    "litellm": _make_client_litellm,
}

API_LLM_BACKENDS: dict[str, dict] = {
    # Example:
    # "myhost": {
    #     "client": "openai",          # which builder to use
    #     "base_url": "https://...",    # OpenAI-compatible endpoint
    #     "api_key_env": "MYHOST_KEY", # env var for the key
    # },
}


def _make_api_llm_client(api_key: str | None = None, **cfg):
    """Build an LLM client from API_LLM_BACKENDS config."""
    builder_name = cfg.get("client", "openai")
    builder = _LLM_CLIENT_BUILDERS.get(builder_name)
    if builder is None:
        raise ValueError(
            f"Unknown LLM client type '{builder_name}'. "
            f"Available: {', '.join(_LLM_CLIENT_BUILDERS)}")
    return builder(api_key, **cfg)


class _LLMBackendsView:
    """Dict-like live view over the LLM backend registries.

    Resolves ``_LLM_CLIENT_BUILDERS`` (built-ins),
    ``PLUGIN_LLM_BACKENDS``, and ``API_LLM_BACKENDS`` on every access,
    so runtime mutations (e.g. adding a new tenant via
    ``API_LLM_BACKENDS["foo"] = {...}``) take effect immediately
    without needing to re-import.
    """

    def _snapshot(self) -> dict:
        return {
            **_LLM_CLIENT_BUILDERS,
            **{n: _make_plugin_loader(p)
               for n, p in PLUGIN_LLM_BACKENDS.items()},
            **{n: partial(_make_api_llm_client, **cfg)
               for n, cfg in API_LLM_BACKENDS.items()},
        }

    def __contains__(self, key):
        return key in self._snapshot()

    def __iter__(self):
        return iter(self._snapshot())

    def __len__(self):
        return len(self._snapshot())

    def get(self, key, default=None):
        return self._snapshot().get(key, default)

    def keys(self):
        return self._snapshot().keys()

    def items(self):
        return self._snapshot().items()

    def values(self):
        return self._snapshot().values()


LLM_BACKENDS = _LLMBackendsView()


def _make_client(
    model: str,
    api_key: str | None = None,
    llm_backend: str | None = None,
):
    """Build an instructor-wrapped async client for the given model.

    Args:
        model: Model ID string. If the model is registered in
            ``MODELS``, its ``config`` dict is passed to the backend
            builder (lets e.g. multiple Azure deployments coexist).
        api_key: Optional API key override.
        llm_backend: Optional provider override. Must be a key in
            ``LLM_BACKENDS``. If None, resolved via the ``MODELS``
            registry or prefix inference.

    Returns:
        An instructor-wrapped async client ready for
        ``client.chat.completions.create()``.
    """
    provider = _get_provider(model, llm_backend)
    fn = LLM_BACKENDS.get(provider)
    if fn is None:
        raise ValueError(
            f"LLM backend '{provider}' not found. "
            f"Available backends: {', '.join(LLM_BACKENDS)}"
        )
    entry = MODELS.get(model, {})
    config = entry.get("config", {})
    return fn(api_key, **config)


def _make_messages(text: str, instructions: str = "") -> list[dict]:
    system = SYSTEM
    if instructions:
        system += "\n\nAdditional instructions:\n" + instructions
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Extract fields from this document:\n\n{text}",
        },
    ]


# --- Single-file extraction ---

async def extract_async(
    pdf_path: str,
    response_model: type[BaseModel],
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    instructions: str = "",
    parser: str = "pymupdf",
    llm_backend: str | None = None,
    text: str | None = None,
    parse_fn=None,
    parser_options: dict | None = None,
) -> BaseModel:
    """Extract structured data from a single PDF.

    Args:
        pdf_path: Path to the PDF file.
        response_model: Pydantic model for structured output.
        model: LLM model ID (default: gpt-4.1-mini).
        api_key: Optional API key override.
        instructions: Additional extraction instructions appended to
            the system prompt.
        parser: Text extraction backend — see ``extract_text()``.
        llm_backend: LLM backend — see ``_make_client()``.
        text: Pre-extracted text (skips PDF parsing if provided).
        parse_fn: Optional async callable(pdf_path, parser)
            -> str. When provided, replaces local text extraction.

    Returns:
        Populated Pydantic model instance.
    """
    _p_opts = parser_options or {}
    mgr = get_manager()
    if text is None:
        if parse_fn is not None:
            text = await mgr.run(
                parse_fn, pdf_path, parser,
            )
        else:
            parser_fn = PARSERS.get(parser)
            if parser_fn and asyncio.iscoroutinefunction(
                parser_fn
            ):
                pages = await mgr.run(
                    parser_fn, pdf_path, **_p_opts,
                )
                text = "\n\n".join(pages)
            else:
                text = await mgr.run_cpu(
                    extract_text, pdf_path, parser,
                    _p_opts,
                )
    if len(text) > TEXT_WARN_THRESHOLD:
        warnings.warn(
            f"Document is large ({len(text):,} chars). "
            "Results may be incomplete. For tabular schemas, "
            "use extract_pages_async() instead.",
            stacklevel=2,
        )
    client = _make_client(model, api_key, llm_backend)
    api_model = _resolve_api_model(model)
    label = os.path.basename(pdf_path)
    for attempt in range(5):
        try:
            async with mgr.api():
                result = await client.chat.completions.create(
                    model=api_model,
                    response_model=response_model,
                    max_retries=2,
                    messages=_make_messages(
                        text, instructions,
                    ),
                    **_model_kwargs(model),
                )
                data = result.model_dump(by_alias=True)
                for msg in _check_extraction_quality(
                    data, text, label=label,
                ):
                    print(msg, flush=True)
                return result
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str
                or "rate" in err_str.lower()
                or "quota" in err_str.lower()
                or "capacity" in err_str.lower()
            )
            if is_rate_limit and attempt < 4:
                wait = 2 ** attempt + 1
                print(
                    f"[petey] rate limited, retrying "
                    f"in {wait}s "
                    f"(attempt {attempt + 1}/5)",
                    flush=True,
                )
                await asyncio.sleep(wait)
                continue
            raise


def extract(
    pdf_path: str,
    response_model: type[BaseModel],
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    instructions: str = "",
    parser: str = "pymupdf",
    llm_backend: str | None = None,
    parser_options: dict | None = None,
) -> BaseModel:
    """Sync wrapper around ``extract_async``. See that function for args."""
    return asyncio.run(
        extract_async(
            pdf_path, response_model,
            model=model, api_key=api_key, instructions=instructions,
            parser=parser, llm_backend=llm_backend,
            parser_options=parser_options,
        )
    )


# --- Schema inference ---

INFER_SCHEMA_SYSTEM = (
    "You analyze PDF documents and suggest extraction schemas.\n"
    "Given sample text from a document, propose a structured "
    "schema with fields to extract.\n\n"
    "Return valid JSON with this structure:\n"
    "{\n"
    '  "name": "short_snake_case_name",\n'
    '  "mode": "query" or "table",\n'
    '  "instructions": "brief notes about the document format",\n'
    '  "fields": {\n'
    '    "field_name": {\n'
    '      "type": "string|number|date|enum",\n'
    '      "description": "what this field contains",\n'
    '      "values": ["only", "for", "enum", "fields"]\n'
    "    }\n"
    "  }\n"
    "}\n\n"
    "Guidelines:\n"
    "- Use table mode for repeating rows/records\n"
    "- Use query mode if the document has one set of fields\n"
    "- Use enum when values come from a fixed set\n"
    "- Use date for dates, number for numbers, string otherwise\n"
    "- Keep field names short, lowercase, snake_case\n"
    "- Include descriptions that help an LLM find the data\n"
    "- For checkbox/range columns, use enum with range labels\n"
    "- Be thorough — include all extractable fields"
)


async def infer_schema_async(
    pdf_path: str,
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    parser: str = "pymupdf",
    llm_backend: str | None = None,
    max_pages: int = 2,
    page_range: str | None = None,
    header_pages: int = 0,
) -> dict:
    """Analyze a PDF and suggest an extraction schema.

    Reads up to ``max_pages`` content pages (plus any header
    pages), sends the text to the LLM, and returns a schema
    spec dict compatible with ``build_model()``.

    When ``page_range`` is set, header pages are taken from
    the start of that range (not the start of the document).

    Args:
        pdf_path: Path to the PDF file.
        model: LLM model ID.
        api_key: Optional API key override.
        parser: Text extraction backend.
        llm_backend: LLM backend override.
        max_pages: Number of content pages to sample (default 2).
        page_range: Optional page range string (e.g. "5-10").
            1-indexed, same format as extraction.
        header_pages: Number of leading pages within the range
            to treat as headers (prepended separately).

    Returns:
        Schema spec dict with name, record_type, fields, etc.
    """
    import json as _json
    import fitz as _fitz

    # Determine which pages we actually need before parsing
    _doc = _fitz.open(pdf_path)
    total = len(_doc)
    _doc.close()

    if page_range:
        indices = _parse_page_range(page_range, total)
    else:
        indices = list(range(total))

    header_indices = indices[:header_pages]
    content_indices = indices[header_pages:]

    # Only parse the pages we need (header + up to max_pages content)
    needed = set(header_indices) | set(content_indices[:max_pages])
    needed_sorted = sorted(needed)

    if needed_sorted:
        subset_path = _subset_pdf(pdf_path, needed_sorted)
        try:
            parsed = extract_text_pages(subset_path, parser)
        finally:
            import os as _os
            _os.unlink(subset_path)
        # Map subset index back to original page index
        page_map = {
            orig: parsed[i]
            for i, orig in enumerate(needed_sorted)
        }
    else:
        page_map = {}

    header_text = ""
    if header_indices:
        header_text = (
            "\n\n---PAGE BREAK---\n\n".join(
                page_map[i] for i in header_indices
            )
            + "\n\n---HEADER END---\n\n"
        )

    sample_pages = [
        page_map[i] for i in content_indices[:max_pages]
    ]
    sample = header_text + "\n\n---PAGE BREAK---\n\n".join(
        sample_pages
    )

    client = _make_client(model, api_key, llm_backend)
    api_model = _resolve_api_model(model)

    # Use raw (unwrapped) client to get plain text back
    raw = getattr(client, "client", client)
    user_msg = (
        "Analyze this document and suggest a schema:"
        f"\n\n{sample}"
    )

    if hasattr(raw, "messages"):
        # Anthropic client
        resp = await raw.messages.create(
            model=api_model,
            system=INFER_SCHEMA_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            temperature=0,
            max_tokens=4096,
        )
        content = resp.content[0].text
    else:
        # OpenAI-compatible client
        resp = await raw.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": INFER_SCHEMA_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            **_model_kwargs(model),
        )
        content = resp.choices[0].message.content

    if not content or not content.strip():
        raise ValueError(
            f"Model '{model}' returned empty response."
        )

    # Parse JSON (handle markdown code blocks)
    text = content.strip()
    if "```" in text:
        text = text.split("```", 1)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        raise ValueError(
            f"Model returned invalid JSON. Response: "
            f"{text[:200]}..."
        )


async def infer_schema_vision_async(
    pdf_path: str,
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    llm_backend: str | None = None,
    max_pages: int = 2,
    page_range: str | None = None,
    header_pages: int = 0,
) -> dict:
    """Analyze a PDF and suggest a schema using vision (images).

    Renders PDF pages as images and sends them directly to the
    LLM, bypassing text parsing entirely.

    Args:
        pdf_path: Path to the PDF file.
        model: LLM model ID (must support vision).
        api_key: Optional API key override.
        llm_backend: LLM backend override.
        max_pages: Number of content pages to sample (default 2).
        page_range: Optional page range string (e.g. "5-10").
        header_pages: Number of leading pages within the range
            to treat as headers.

    Returns:
        Schema spec dict with name, record_type, fields, etc.
    """
    import json as _json
    import base64 as _b64
    import fitz as _fitz

    doc = _fitz.open(pdf_path)
    total = len(doc)

    if page_range:
        indices = _parse_page_range(page_range, total)
    else:
        indices = list(range(total))

    header_indices = indices[:header_pages]
    content_indices = indices[header_pages:]
    needed = list(header_indices) + list(
        content_indices[:max_pages]
    )

    # Render pages to PNG base64
    images = []
    for idx in needed:
        page = doc[idx]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        b64 = _b64.b64encode(img_bytes).decode()
        is_header = idx in header_indices
        images.append((idx, b64, is_header))
    doc.close()

    client = _make_client(model, api_key, llm_backend)
    api_model = _resolve_api_model(model)
    raw = getattr(client, "client", client)

    prompt_text = "Analyze this document and suggest a schema."
    if header_indices:
        prompt_text += (
            " The first image(s) are header pages that "
            "provide context (column names, metadata)."
        )

    if hasattr(raw, "messages"):
        # Anthropic — images as content blocks
        content_blocks = [{"type": "text", "text": prompt_text}]
        for idx, b64, is_header in images:
            label = f"header page {idx+1}" if is_header else f"page {idx+1}"
            content_blocks.append({
                "type": "text", "text": f"[{label}]",
            })
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64,
                },
            })
        resp = await raw.messages.create(
            model=api_model,
            system=INFER_SCHEMA_SYSTEM,
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0,
            max_tokens=4096,
        )
        content = resp.content[0].text
    else:
        # OpenAI — images as content parts
        content_parts = [{"type": "text", "text": prompt_text}]
        for idx, b64, is_header in images:
            label = f"header page {idx+1}" if is_header else f"page {idx+1}"
            content_parts.append({
                "type": "text", "text": f"[{label}]",
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                },
            })
        resp = await raw.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": INFER_SCHEMA_SYSTEM},
                {"role": "user", "content": content_parts},
            ],
            **_model_kwargs(model),
        )
        content = resp.choices[0].message.content

    if not content or not content.strip():
        raise ValueError(
            f"Model '{model}' returned empty response. "
            "It may not support vision or the input was "
            "too large. Try a different model."
        )

    text = content.strip()
    if "```" in text:
        text = text.split("```", 1)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        raise ValueError(
            f"Model returned invalid JSON. Response: "
            f"{text[:200]}..."
        )


def infer_schema(
    pdf_path: str,
    **kwargs,
) -> dict:
    """Sync wrapper around ``infer_schema_async``."""
    return asyncio.run(
        infer_schema_async(pdf_path, **kwargs)
    )


# --- Page-chunked extraction ---

def _parse_page_range(spec: str, total_pages: int) -> list[int]:
    """Parse a page range string into sorted 0-indexed page indices.

    Examples: "2-5" -> [1,2,3,4], "1,3,5-7" -> [0,2,4,5,6]
    """
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start) - 1, int(end)))
        else:
            indices.add(int(part) - 1)
    return sorted(i for i in indices if 0 <= i < total_pages)


def _subset_pdf(pdf_path: str, page_indices: list[int]) -> str:
    """Create a temporary PDF containing only the given pages.

    Args:
        pdf_path: Path to the source PDF.
        page_indices: 0-indexed page numbers to include.

    Returns:
        Path to a temporary PDF file. Caller is responsible
        for cleanup.
    """
    doc = fitz.open(pdf_path)
    subset = fitz.open()
    for i in page_indices:
        subset.insert_pdf(doc, from_page=i, to_page=i)
    doc.close()
    tmp = tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False,
    )
    subset.save(tmp.name)
    subset.close()
    tmp.close()
    return tmp.name


async def extract_pages_async(
    pdf_path: str,
    response_model: type[BaseModel],
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    instructions: str = "",
    pages_per_chunk: int = 1,
    concurrency: int = 10,
    on_result=None,
    on_parse=None,
    parser: str = "pymupdf",
    llm_backend: str | None = None,
    header_pages: int = 0,
    page_range: str | None = None,
    parse_multiplier: int = 5,
    parse_fn=None,
    parser_options: dict | None = None,
) -> list[dict]:
    """Split a PDF into page chunks and extract each concurrently.

    Args:
        pdf_path: Path to the PDF file.
        response_model: Pydantic model for structured output.
        model: LLM model ID (default: gpt-4.1-mini).
        api_key: Optional API key override.
        instructions: Additional extraction instructions.
        pages_per_chunk: Number of pages per chunk (default 1).
        concurrency: Max concurrent API calls.
        on_result: Optional callback(chunk_label, data_dict) called
            as each chunk's LLM extraction completes.
        on_parse: Optional callback(chunk_label, total_chunks)
            called as each chunk's text is parsed from the PDF.
        parser: Text extraction backend.
        llm_backend: LLM backend — see ``_make_client()``.
        header_pages: Number of leading pages to treat as a
            header. Their text is prepended to every chunk.
        page_range: Optional page range string (e.g. "2-5" or
            "1,3,5-7"). 1-indexed.
        parse_fn: Optional async callable(pdf_path, page_index,
            parser) -> str. Replaces local page-level extraction.
        parser_options: Extra kwargs passed to the parser.

    Returns:
        List of result dicts (with _page and optionally _error).
    """
    if parse_multiplier != 5:
        warnings.warn(
            "parse_multiplier is deprecated and ignored. "
            "CPU and API concurrency are now managed "
            "separately by ConcurrencyManager.",
            DeprecationWarning,
            stacklevel=2,
        )

    _p_opts = parser_options or {}

    # Resolve parser from registry
    parser_fn = PARSERS.get(parser)
    if parser_fn is None:
        raise ValueError(
            f"Parser '{parser}' not found. "
            f"Available: {', '.join(PARSERS)}"
        )
    parser_is_async = asyncio.iscoroutinefunction(
        parser_fn
    )

    # Get page count without parsing everything
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    print(
        f"[petey] {total_pages} pages, "
        f"parser={parser}, "
        f"page_range={page_range}",
        flush=True,
    )

    mgr = get_manager()
    mgr.configure(api_limit=concurrency)

    # Parse header pages upfront (small, fast)
    header_text = ""
    if header_pages > 0:
        h_indices = list(
            range(min(header_pages, total_pages))
        )
        h_path = _subset_pdf(pdf_path, h_indices)
        try:
            if parser_is_async:
                h_pages = await mgr.run(
                    parser_fn, h_path, **_p_opts,
                )
            else:
                h_pages = await mgr.run_cpu(
                    parser_fn, h_path, **_p_opts,
                )
            header_text = "\n\n".join(h_pages)
        finally:
            os.unlink(h_path)

    # Determine content page indices
    if page_range:
        content_indices = [
            i for i in _parse_page_range(
                page_range, total_pages,
            )
            if i >= header_pages
        ]
    else:
        content_indices = list(
            range(header_pages, total_pages)
        )

    # If all pages are headers (e.g. 1-page PDF with
    # header_pages=1) or header pages outnumber content pages,
    # include header pages as content too so their data gets
    # extracted (not just prepended as context).
    if total_pages > 0 and len(content_indices) <= header_pages:
        content_indices = list(range(total_pages))
        header_text = ""

    # Build chunk index groups (for pages_per_chunk > 1)
    chunk_groups = []
    for chunk_start in range(
        0, len(content_indices), pages_per_chunk,
    ):
        chunk_groups.append(
            content_indices[
                chunk_start:chunk_start + pages_per_chunk
            ]
        )

    print(
        f"[petey] {len(chunk_groups)} chunks to process",
        flush=True,
    )
    client = _make_client(model, api_key, llm_backend)
    api_model = _resolve_api_model(model)
    results = [None] * len(chunk_groups)
    if not chunk_groups:
        print(
            f"[petey] no content pages to process "
            f"(header_pages={header_pages}, "
            f"total={total_pages})",
            flush=True,
        )
        return []

    async def _parse_chunk(idx_slice):
        """Subset the PDF and parse via the registry."""
        chunk_path = _subset_pdf(pdf_path, idx_slice)
        try:
            if parser_is_async:
                pages = await mgr.run(
                    parser_fn, chunk_path, **_p_opts,
                )
            else:
                pages = await mgr.run_cpu(
                    parser_fn, chunk_path, **_p_opts,
                )
            return "\n\n".join(pages)
        finally:
            os.unlink(chunk_path)

    async def _parse_and_extract(idx, idx_slice):
        print(
            f"[petey] chunk {idx}: parsing pages "
            f"{idx_slice}", flush=True,
        )
        text = await _parse_chunk(idx_slice)
        if header_text:
            text = header_text + "\n\n" + text
        start = idx_slice[0] + 1
        end = idx_slice[-1] + 1
        label = (
            f"p{start}" if start == end
            else f"p{start}-{end}"
        )
        print(
            f"[petey] chunk {idx} ({label}): "
            f"parsed, sending to LLM", flush=True,
        )
        if on_parse:
            on_parse(label, len(chunk_groups))

        async with mgr.api():
            data = None
            for attempt in range(5):
                try:
                    result = (
                        await client.chat.completions.create(
                            model=api_model,
                            response_model=response_model,
                            max_retries=2,
                            messages=_make_messages(
                                text, instructions,
                            ),
                            **_model_kwargs(model),
                        )
                    )
                    data = result.model_dump(by_alias=True)
                    data["_page"] = label
                    for msg in _check_extraction_quality(
                        data, text, label=label,
                    ):
                        print(msg, flush=True)
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = (
                        "429" in err_str
                        or "rate" in err_str.lower()
                        or "quota" in err_str.lower()
                        or "capacity" in err_str.lower()
                    )
                    if is_rate_limit and attempt < 4:
                        wait = 2 ** attempt + 1
                        print(
                            f"[petey] {label}: rate "
                            f"limited, retrying in "
                            f"{wait}s (attempt "
                            f"{attempt + 1}/5)",
                            flush=True,
                        )
                        await asyncio.sleep(wait)
                        continue
                    data = {
                        "_page": label,
                        "_error": err_str,
                    }
                    break
            results[idx] = data
            if on_result:
                on_result(label, data)

    await asyncio.gather(
        *[
            _parse_and_extract(i, idx_slice)
            for i, idx_slice in enumerate(chunk_groups)
        ]
    )

    # Deduplicate results, keeping first occurrence per unique row
    seen = set()
    deduped = []
    dupes = 0
    for r in results:
        if r is None or r.get("_error"):
            deduped.append(r)
            continue
        # Build key from all fields except _page and _error
        key_fields = tuple(
            sorted(
                (k, str(v)) for k, v in r.items()
                if k not in ("_page", "_error")
            )
        )
        if key_fields in seen:
            dupes += 1
            continue
        seen.add(key_fields)
        deduped.append(r)
    if dupes:
        print(f"[petey] deduplicated {dupes} duplicate rows ({len(results)} -> {len(deduped)})", flush=True)
    return deduped


# --- Batch extraction ---

async def extract_batch(
    pdf_paths: list[str],
    response_model: type[BaseModel],
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    instructions: str = "",
    concurrency: int = 10,
    on_result=None,
    parser: str = "pymupdf",
    llm_backend: str | None = None,
    parse_fn=None,
    parser_options: dict | None = None,
) -> list[dict]:
    """Extract from multiple PDFs concurrently.

    Args:
        pdf_paths: List of PDF file paths.
        response_model: Pydantic model for structured output.
        model: LLM model ID (default: gpt-4.1-mini).
        api_key: Optional API key override.
        instructions: Additional extraction instructions.
        concurrency: Max concurrent API calls.
        on_result: Optional callback(path, data_dict) called
            as each file completes.
        parser: Text extraction backend.
        llm_backend: LLM backend — see ``_make_client()``.
        parse_fn: Optional async callable(pdf_path, parser)
            -> str. Replaces local text extraction.
        parser_options: Extra kwargs passed to the parser.

    Returns:
        List of result dicts (with _source_file and
        optionally _error).
    """
    _p_opts = parser_options or {}
    mgr = get_manager()
    mgr.configure(api_limit=concurrency)
    client = _make_client(model, api_key, llm_backend)
    api_model = _resolve_api_model(model)
    results = []

    parser_fn = PARSERS.get(parser)
    parser_is_async = (
        parser_fn is not None
        and asyncio.iscoroutinefunction(parser_fn)
    )

    async def _process(path: str):
        try:
            if parse_fn is not None:
                text = await mgr.run(
                    parse_fn, path, parser,
                )
            elif parser_is_async:
                pages = await mgr.run(
                    parser_fn, path, **_p_opts,
                )
                text = "\n\n".join(pages)
            else:
                text = await mgr.run_cpu(
                    extract_text, path, parser,
                    _p_opts,
                )
            async with mgr.api():
                result = (
                    await client.chat.completions.create(
                        model=api_model,
                        response_model=response_model,
                        max_retries=2,
                        messages=_make_messages(
                            text, instructions,
                        ),
                        **_model_kwargs(model),
                    )
                )
            data = result.model_dump(by_alias=True)
            fname = os.path.basename(path)
            data["_source_file"] = fname
            for msg in _check_extraction_quality(
                data, text, label=fname,
            ):
                print(msg, flush=True)
        except Exception as e:
            data = {
                "_source_file": os.path.basename(path),
                "_error": str(e),
            }
        results.append(data)
        if on_result:
            on_result(path, data)

    await asyncio.gather(
        *[_process(p) for p in pdf_paths]
    )
    return results
