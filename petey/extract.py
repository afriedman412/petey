"""
Core PDF extraction. No web dependencies.
Configure via environment variables or pass explicitly.

Pipeline architecture::

    PDF
     └─ TextExtractor      (pymupdf | pdfplumber | tabula | marker)
          └─ OCRBackend    (tesseract | mistral | chandra | none)
               └─ LLMBackend  (openai | anthropic | litellm)
                    └─ Output  (csv | json | jsonl)

Each layer is swappable via parameters on the public functions.
"""
import asyncio
import base64
from petey.concurrency import get_manager
from functools import partial
import json
import os
import tempfile
import warnings

import httpx

import fitz
import pymupdf4llm
import instructor
from pydantic import BaseModel
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

SYSTEM = (
    "You extract structured data from documents. "
    "Use null for missing or unreadable values."
)

TEXT_WARN_THRESHOLD = 50_000


# --- PDF text extraction backends ---

OCR_THRESHOLD = 100  # chars below which pymupdf output is treated as empty


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


def _extract_text_pages_tables(pdf_path: str) -> list[str]:
    """Extract per-page text using PyMuPDF's table detection.

    Detects bordered tables and converts them to TSV.  Falls back to plain
    text for pages without tables.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        found = page.find_tables()
        if found.tables:
            parts = []
            for table in found.tables:
                rows = table.extract()
                tsv = "\n".join(
                    "\t".join(
                        str(cell) if cell else "" for cell in row
                    )
                    for row in rows
                    if any(cell for cell in row)
                )
                if tsv:
                    parts.append(tsv)
            text = (
                "\n\n".join(parts) if parts
                else page.get_text("text")
            )
        else:
            text = page.get_text("text")
        pages.append(text)
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


def _extract_text_pages_tabula(pdf_path: str) -> list[str]:
    """Extract per-page text using tabula-py table detection.

    Uses tabula-py to read tables as DataFrames, converts to TSV strings.
    Falls back to pymupdf4llm markdown for pages with no detected tables.
    Requires Java to be installed.
    """
    try:
        import tabula
    except ImportError:
        raise ImportError(
            "tabula-py is required for parser='tabula'. "
            "Install it with: pip install petey[tabula]"
        )
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    pages = []
    for page_num in range(1, n_pages + 1):
        try:
            dfs = tabula.read_pdf(
                pdf_path, pages=page_num, multiple_tables=True,
                silent=True,
            )
            if dfs:
                parts = []
                for df in dfs:
                    tsv = df.to_csv(sep="\t", index=False)
                    if tsv.strip():
                        parts.append(tsv.strip())
                if parts:
                    pages.append("\n\n".join(parts))
                    continue
        except Exception:
            pass
        # Fallback to pymupdf4llm markdown
        fallback = pymupdf4llm.to_markdown(pdf_path, pages=[page_num - 1])
        pages.append(fallback)
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
#   timeout        — max seconds to wait for poll (default: 240)

API_PARSERS: dict[str, dict] = {
    "marker": {
        "endpoint": "https://www.datalab.to/api/v1/marker",
        "api_key_env": "DATALAB_API_KEY",
        "auth_header": "X-API-Key",
        "params": {"output_format": "markdown"},
        "response_key": "markdown",
        "poll": True,
    },
    "llamaparse": {
        "endpoint": "https://api.cloud.llamaindex.ai/api/parsing/upload",
        "api_key_env": "LLAMA_CLOUD_API_KEY",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer",
        "response_key": "markdown",
        "poll": True,
        "poll_check_key": "id",
        "poll_url_template": (
            "https://api.cloud.llamaindex.ai"
            "/api/parsing/job/{id}/result/markdown"
        ),
        "poll_status_key": "status",
        "poll_done_value": "SUCCESS",
    },
}

API_OCR_BACKENDS: dict[str, dict] = {
    "chandra": {
        "endpoint": "https://www.datalab.to/api/v1/chandra",
        "api_key_env": "DATALAB_API_KEY",
        "auth_header": "X-API-Key",
        "params": {"output_format": "markdown"},
        "response_key": "markdown",
        "poll": True,
    },
}


def _resolve_response(data: dict, key_path: str) -> str:
    """Extract a value from a nested dict using a dot-separated path.

    ``"markdown"`` → ``data["markdown"]``
    ``"result.text"`` → ``data["result"]["text"]``
    """
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
        result = resp.json()

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
            poll_url_template = cfg.get("poll_url_template")
            if poll_url_template:
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
        raise ValueError(f"{env_var} environment variable is required.")
    return api_key


async def _parse_pdf_via_api(pdf_path: str, cfg: dict) -> list[str]:
    """Parse a full PDF by uploading it to a remote API."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    text = await _api_post(
        cfg, pdf_bytes, "document.pdf", "application/pdf",
    )
    return [text] if text else [""]


async def _ocr_page_via_api(page, cfg: dict) -> str:
    """OCR a single fitz page by uploading it to a remote API."""
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    return await _api_post(
        cfg, img_bytes, "page.png", "image/png",
    )


# --- Parser registry ---
# Each parser is a callable: (pdf_path: str) -> list[str]
# Local parsers are plain functions; API parsers are built from config.

PARSERS = {
    "pymupdf": _extract_text_pages_pymupdf,
    "tables": _extract_text_pages_tables,
    "pdfplumber": _extract_text_pages_pdfplumber,
    "tabula": _extract_text_pages_tabula,
    **{name: partial(_parse_pdf_via_api, cfg=cfg)
       for name, cfg in API_PARSERS.items()},
}


def extract_text(
    pdf_path: str,
    parser: str = "pymupdf",
    ocr_fallback: bool = False,
    ocr_backend: str = "none",
) -> str:
    """Extract all text from a PDF as a single string.

    Args:
        pdf_path: Path to the PDF file.
        parser: Text extraction backend.
            "pymupdf"    — markdown via pymupdf4llm (default)
            "tables"     — PyMuPDF table detection → TSV
            "pdfplumber" — layout-preserving spatial extraction
            "tabula"     — tabula-py table extraction → TSV (requires Java)
        ocr_fallback: Deprecated. Use ``ocr_backend="tesseract"`` instead.
        ocr_backend: OCR fallback when text layer is empty/short.
            "none"       — no OCR (default)
            "tesseract"  — local pytesseract (requires tesseract binary)
            "mistral"    — Mistral OCR API (requires MISTRAL_API_KEY)

    Returns:
        Extracted text as a single string.
    """
    pages = extract_text_pages(pdf_path, parser, ocr_fallback, ocr_backend)
    return "\n\n".join(pages)


def extract_text_pages(
    pdf_path: str,
    parser: str = "pymupdf",
    ocr_fallback: bool = False,
    ocr_backend: str = "none",
) -> list[str]:
    """Extract text from each page of a PDF separately.

    Args:
        pdf_path: Path to the PDF file.
        parser: Text extraction backend — see ``PARSERS`` registry.
        ocr_fallback: Deprecated. Use ``ocr_backend="tesseract"`` instead.
        ocr_backend: OCR fallback when text layer is empty/short —
            see ``OCR_BACKENDS`` registry.

    Returns:
        List of strings, one per page.
    """
    ocr_backend = _resolve_ocr_backend(ocr_fallback, ocr_backend)

    fn = PARSERS.get(parser)
    if fn is None:
        raise ValueError(
            f"Parser '{parser}' not found. "
            f"Available parsers: {', '.join(PARSERS)}"
        )
    # When an external OCR backend is requested and the user hasn't
    # explicitly chosen a parser, use pdfplumber instead of pymupdf4llm
    # so that scanned pages are left empty for the external OCR to handle
    # (pymupdf4llm runs its own built-in OCR which can't be fully disabled).
    if parser == "pymupdf" and ocr_backend != "none":
        fn = PARSERS["pdfplumber"]
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
        pages = asyncio.run(fn(pdf_path))
    else:
        pages = fn(pdf_path)

    # Apply OCR fallback to pages with insufficient text
    if ocr_backend != "none":
        doc = None
        for i, text in enumerate(pages):
            if len(text.strip()) < OCR_THRESHOLD:
                if doc is None:
                    doc = fitz.open(pdf_path)
                pages[i] = _ocr_page(doc[i], ocr_backend, pdf_path)

    return pages


def _resolve_ocr_backend(ocr_fallback: bool, ocr_backend: str) -> str:
    """Normalise the deprecated ``ocr_fallback`` flag into an ocr_backend."""
    if ocr_fallback and ocr_backend == "none":
        return "tesseract"
    return ocr_backend


def _ocr_page_tesseract(page) -> str:
    """OCR a single fitz page using pytesseract."""
    import pytesseract
    from PIL import Image
    import io
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _ocr_page_mistral(page) -> str:
    """OCR a single fitz page using Mistral's OCR API.

    Renders the page to a PNG, base64-encodes it, and sends it to
    Mistral's ``mistral-ocr-latest`` model.

    Requires ``MISTRAL_API_KEY`` environment variable or will raise
    ``ValueError``.
    """
    try:
        from mistralai import Mistral
    except ImportError:
        raise ImportError(
            "mistralai is required for ocr_backend='mistral'. "
            "Install it with: pip install petey[mistral-ocr]"
        )
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "Mistral OCR requires MISTRAL_API_KEY environment variable."
        )
    pix = page.get_pixmap(dpi=200)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    data_url = f"data:image/png;base64,{img_b64}"

    client = Mistral(api_key=api_key)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "image_url", "image_url": data_url},
    )
    # Response contains pages with markdown text content
    parts = []
    for p in response.pages:
        if p.markdown:
            parts.append(p.markdown)
    return "\n\n".join(parts) if parts else ""


# --- OCR backend registry ---
# Each backend is a callable: (page: fitz.Page) -> str
# Local backends are plain functions; API backends are built from config.

OCR_BACKENDS = {
    "tesseract": _ocr_page_tesseract,
    "mistral": _ocr_page_mistral,
    **{name: partial(_ocr_page_via_api, cfg=cfg)
       for name, cfg in API_OCR_BACKENDS.items()},
}


def _ocr_page(
    page, ocr_backend: str = "tesseract",
    pdf_path: str | None = None,
) -> str:
    """OCR a single fitz page.

    Args:
        page: A fitz page object.
        ocr_backend: OCR backend — see ``OCR_BACKENDS`` registry.
        pdf_path: Path to the source PDF (used by Mistral OCR).

    Returns:
        Extracted text from the page.
    """
    fn = OCR_BACKENDS.get(ocr_backend)
    if fn is None:
        raise ValueError(
            f"OCR backend '{ocr_backend}' not found. "
            f"Available: {', '.join(OCR_BACKENDS)}"
        )
    if asyncio.iscoroutinefunction(fn):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError(
                f"OCR backend '{ocr_backend}' is async. "
                f"Use extract_async() or "
                f"extract_pages_async() instead."
            )
        return asyncio.run(fn(page))
    return fn(page)


def _ocr_full(doc, pdf_path: str, ocr_backend: str) -> str:
    """OCR all pages of an open fitz document."""
    return "\n\n".join(
        _ocr_page(page, ocr_backend, pdf_path) for page in doc
    )


def _ocr_full_doc(pdf_path: str) -> str:
    """OCR all pages of a PDF using tesseract."""
    doc = fitz.open(pdf_path)
    pages = [_ocr_page_tesseract(page) for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _ocr_single_page(pdf_path: str, page_index: int) -> str:
    """OCR a single page by index. Picklable for ProcessPoolExecutor."""
    doc = fitz.open(pdf_path)
    text = _ocr_page_tesseract(doc[page_index])
    doc.close()
    return text

# --- LLM client ---

def _get_provider(model: str, llm_backend: str | None = None) -> str:
    """Determine which LLM provider to use.

    Args:
        model: Model ID string (e.g. "gpt-4.1-mini", "claude-sonnet-4-6").
        llm_backend: Explicit override. One of:
            "openai"    — direct OpenAI client
            "anthropic" — direct Anthropic client
            "litellm"   — unified router (gemini/, mistral/, ollama/, etc.)
            None        — auto-detect from model string (default)

    Returns:
        Provider string: "openai", "anthropic", or "litellm".
    """
    if llm_backend is not None:
        return llm_backend
    if model.startswith("claude"):
        return "anthropic"
    # Models that need litellm routing
    litellm_prefixes = (
        "gemini/", "mistral/", "ollama/", "ollama_chat/",
        "bedrock/", "vertex_ai/", "cohere/", "replicate/",
        "huggingface/", "together_ai/", "openrouter/",
    )
    if any(model.startswith(p) for p in litellm_prefixes):
        return "litellm"
    return "openai"


def _make_client_openai(api_key: str | None = None, **kwargs):
    """Build an instructor-wrapped OpenAI async client."""
    base_url = kwargs.get("base_url")
    key = api_key or os.environ.get(
        kwargs.get("api_key_env", "OPENAI_API_KEY"))
    client_kwargs = {"api_key": key}
    if base_url:
        client_kwargs["base_url"] = base_url
    return instructor.from_openai(AsyncOpenAI(**client_kwargs))


def _make_client_anthropic(api_key: str | None = None, **kwargs):
    """Build an instructor-wrapped Anthropic async client."""
    key = api_key or os.environ.get(
        kwargs.get("api_key_env", "ANTHROPIC_API_KEY"))
    return instructor.from_anthropic(
        AsyncAnthropic(api_key=key), max_tokens=16384,
    )


def _make_client_litellm(api_key: str | None = None, **kwargs):
    """Build an instructor-wrapped litellm client."""
    from litellm import acompletion
    return instructor.from_litellm(acompletion)


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


LLM_BACKENDS = {
    **_LLM_CLIENT_BUILDERS,
    **{name: partial(_make_api_llm_client, **cfg)
       for name, cfg in API_LLM_BACKENDS.items()},
}


def _make_client(
    model: str,
    api_key: str | None = None,
    llm_backend: str | None = None,
):
    """Build an instructor-wrapped async client for the given model.

    Args:
        model: Model ID string. Determines provider auto-detection.
        api_key: Optional API key override.
        llm_backend: Override auto-detection.
            "openai"    — direct OpenAI client
            "anthropic" — direct Anthropic client
            "litellm"   — unified router (supports 100+ providers)
            None        — auto-detect from model string (default)

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
    return fn(api_key)


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
    ocr_fallback: bool = False,
    ocr_backend: str = "none",
    llm_backend: str | None = None,
    text: str | None = None,
    parse_fn=None,
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
        ocr_fallback: Deprecated. Use ``ocr_backend`` instead.
        ocr_backend: OCR backend — see ``extract_text()``.
        llm_backend: LLM backend — see ``_make_client()``.
        text: Pre-extracted text (skips PDF parsing if provided).
        parse_fn: Optional async callable(pdf_path, parser, ocr_backend)
            -> str. When provided, replaces local text extraction.

    Returns:
        Populated Pydantic model instance.
    """
    mgr = get_manager()
    if text is None:
        if parse_fn is not None:
            text = await mgr.run(
                parse_fn, pdf_path, parser, ocr_backend,
            )
        else:
            # Look up the parser to dispatch correctly:
            # async (API) → mgr.run, sync (local) → mgr.run_cpu
            parser_fn = PARSERS.get(parser)
            if parser_fn and asyncio.iscoroutinefunction(
                parser_fn
            ):
                pages = await mgr.run(
                    parser_fn, pdf_path,
                )
                text = "\n\n".join(pages)
            else:
                text = await mgr.run_cpu(
                    extract_text, pdf_path, parser,
                    ocr_fallback, ocr_backend,
                )
    if len(text) > TEXT_WARN_THRESHOLD:
        warnings.warn(
            f"Document is large ({len(text):,} chars). "
            "Results may be incomplete. For tabular schemas, "
            "use extract_pages_async() instead.",
            stacklevel=2,
        )
    client = _make_client(model, api_key, llm_backend)
    ocr_retried = False
    for attempt in range(5):
        try:
            async with mgr.api():
                return await client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    max_retries=2,
                    messages=_make_messages(
                        text, instructions,
                    ),
                    temperature=0,
                    max_tokens=4096,
                )
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
            if not ocr_retried and pdf_path:
                ocr_retried = True
                print(
                    "[petey] extraction failed, "
                    "retrying with OCR fallback",
                    flush=True,
                )
                try:
                    ocr_text = await mgr.run_cpu(
                        _ocr_full_doc, pdf_path,
                    )
                    if (ocr_text
                            and len(ocr_text.strip()) > 50):
                        text = ocr_text
                        continue
                except Exception:
                    pass
            raise


def extract(
    pdf_path: str,
    response_model: type[BaseModel],
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    instructions: str = "",
    parser: str = "pymupdf",
    ocr_fallback: bool = False,
    ocr_backend: str = "none",
    llm_backend: str | None = None,
) -> BaseModel:
    """Sync wrapper around ``extract_async``. See that function for args."""
    return asyncio.run(
        extract_async(
            pdf_path, response_model,
            model=model, api_key=api_key, instructions=instructions,
            parser=parser, ocr_fallback=ocr_fallback,
            ocr_backend=ocr_backend, llm_backend=llm_backend,
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
    ocr_fallback: bool = False,
    ocr_backend: str = "none",
    llm_backend: str | None = None,
    max_pages: int = 2,
) -> dict:
    """Analyze a PDF and suggest an extraction schema.

    Reads the first ``max_pages`` pages, sends the text to
    the LLM, and returns a schema spec dict compatible with
    ``build_model()``.

    Args:
        pdf_path: Path to the PDF file.
        model: LLM model ID.
        api_key: Optional API key override.
        parser: Text extraction backend.
        ocr_fallback: Fall back to OCR for low-text pages.
        ocr_backend: OCR backend name.
        llm_backend: LLM backend override.
        max_pages: Number of pages to sample (default 2).

    Returns:
        Schema spec dict with name, record_type, fields, etc.
    """
    import json as _json

    pages = extract_text_pages(
        pdf_path, parser,
        ocr_fallback=ocr_fallback, ocr_backend=ocr_backend,
    )
    sample_pages = pages[:max_pages]
    sample = "\n\n---PAGE BREAK---\n\n".join(sample_pages)

    client = _make_client(model, api_key, llm_backend)

    # Use raw (unwrapped) client to get plain text back
    raw = getattr(client, "client", client)
    user_msg = (
        "Analyze this document and suggest a schema:"
        f"\n\n{sample}"
    )
    resp = await raw.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INFER_SCHEMA_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=4096,
    )

    # Extract text content from response
    if hasattr(resp, "content"):
        content = resp.content[0].text
    else:
        content = resp.choices[0].message.content

    # Parse JSON (handle markdown code blocks)
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return _json.loads(text)


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
    ocr_backend: str = "none",
    llm_backend: str | None = None,
    header_pages: int = 0,
    page_range: str | None = None,
    parse_multiplier: int = 5,
    parse_fn=None,
) -> list[dict]:
    """Split a PDF into page chunks and extract each concurrently.

    Parsing and LLM extraction are pipelined: each page is parsed
    individually and its LLM call is launched immediately, so parsing
    page N and extracting page N-1 happen in parallel.

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
        on_parse: Optional callback(chunk_label, total_chunks) called
            as each chunk's text is parsed from the PDF.
        parser: Text extraction backend — see ``extract_text()``.
        ocr_backend: OCR backend — see ``extract_text()``.
        llm_backend: LLM backend — see ``_make_client()``.
        header_pages: Number of leading pages to treat as a header.
            Their text is prepended to every chunk for context.
        page_range: Optional page range string (e.g. "2-5" or
            "1,3,5-7"). 1-indexed. If omitted, all non-header
            pages are processed.
        parse_fn: Optional async callable(pdf_path, page_index, parser,
            ocr_backend) -> str. When provided, replaces local
            page-level text extraction (no ProcessPoolExecutor used).

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
                    parser_fn, h_path,
                )
            else:
                h_pages = await mgr.run_cpu(
                    parser_fn, h_path,
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
    # header_pages=1), treat header pages as content too
    if not content_indices and total_pages > 0:
        content_indices = list(range(total_pages))

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
                    parser_fn, chunk_path,
                )
            else:
                pages = await mgr.run_cpu(
                    parser_fn, chunk_path,
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
            ocr_retried = False
            for attempt in range(5):
                try:
                    result = (
                        await client.chat.completions.create(
                            model=model,
                            response_model=response_model,
                            max_retries=2,
                            messages=_make_messages(
                                text, instructions,
                            ),
                            temperature=0,
                        )
                    )
                    data = result.model_dump()
                    data["_page"] = label
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
                    if not ocr_retried:
                        ocr_retried = True
                        print(
                            f"[petey] {label}: "
                            f"extraction failed "
                            f"({err_str[:80]}), "
                            f"retrying with OCR",
                            flush=True,
                        )
                        try:
                            ocr_pages = (
                                await asyncio.gather(*[
                                    mgr.run_cpu(
                                        _ocr_single_page,
                                        pdf_path, i,
                                    )
                                    for i in idx_slice
                                ])
                            )
                            ocr_text = "\n\n".join(
                                ocr_pages
                            )
                            if (
                                ocr_text
                                and len(
                                    ocr_text.strip()
                                ) > 50
                            ):
                                text = (
                                    (
                                        header_text
                                        + "\n\n"
                                        + ocr_text
                                    )
                                    if header_text
                                    else ocr_text
                                )
                                continue
                        except Exception as ocr_err:
                            print(
                                f"[petey] {label}: "
                                f"OCR fallback also "
                                f"failed: {ocr_err}",
                                flush=True,
                            )
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
    ocr_fallback: bool = False,
    ocr_backend: str = "none",
    llm_backend: str | None = None,
    parse_fn=None,
) -> list[dict]:
    """Extract from multiple PDFs concurrently.

    Args:
        pdf_paths: List of PDF file paths.
        response_model: Pydantic model for structured output.
        model: LLM model ID (default: gpt-4.1-mini).
        api_key: Optional API key override.
        instructions: Additional extraction instructions.
        concurrency: Max concurrent API calls.
        on_result: Optional callback(path, data_dict) called as
            each file completes.
        parser: Text extraction backend — see ``extract_text()``.
        ocr_fallback: Deprecated. Use ``ocr_backend`` instead.
        ocr_backend: OCR backend — see ``extract_text()``.
        llm_backend: LLM backend — see ``_make_client()``.
        parse_fn: Optional async callable(pdf_path, parser,
            ocr_backend) -> str. When provided, replaces local
            text extraction.

    Returns:
        List of result dicts (with _source_file and optionally
        _error).
    """
    mgr = get_manager()
    mgr.configure(api_limit=concurrency)
    client = _make_client(model, api_key, llm_backend)
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
                    parse_fn, path, parser, ocr_backend,
                )
            elif parser_is_async:
                pages = await mgr.run(
                    parser_fn, path,
                )
                text = "\n\n".join(pages)
            else:
                text = await mgr.run_cpu(
                    extract_text, path, parser,
                    ocr_fallback, ocr_backend,
                )
            async with mgr.api():
                result = (
                    await client.chat.completions.create(
                        model=model,
                        response_model=response_model,
                        max_retries=2,
                        messages=_make_messages(
                            text, instructions,
                        ),
                        temperature=0,
                    )
                )
            data = result.model_dump()
            data["_source_file"] = os.path.basename(path)
        except Exception as e:
            data = {
                "_source_file": os.path.basename(path),
                "_error": str(e),
            }
        results.append(data)
        if on_result:
            on_result(path, data)

    await asyncio.gather(*[_process(p) for p in pdf_paths])
    return results
