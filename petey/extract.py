"""
Core PDF extraction. No web dependencies.
Configure via environment variables or pass explicitly.

Pipeline architecture::

    PDF
     └─ TextExtractor      (pymupdf | pdfplumber | tabula)
          └─ OCRBackend    (tesseract | mistral | none)
               └─ LLMBackend  (openai | anthropic | litellm)
                    └─ Output  (csv | json | jsonl)

Each layer is swappable via parameters on the public functions.
"""
import asyncio
import base64
import os
import warnings

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
    chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    return [chunk["text"] for chunk in chunks]


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


# --- Parser registry ---
# Each parser is a callable: (pdf_path: str) -> list[str]

PARSERS = {
    "pymupdf": _extract_text_pages_pymupdf,
    "tables": _extract_text_pages_tables,
    "pdfplumber": _extract_text_pages_pdfplumber,
    "tabula": _extract_text_pages_tabula,
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

OCR_BACKENDS = {
    "tesseract": _ocr_page_tesseract,
    "mistral": _ocr_page_mistral,
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
            f"Available backends: {', '.join(OCR_BACKENDS)}"
        )
    return fn(page)


def _ocr_full(doc, pdf_path: str, ocr_backend: str) -> str:
    """OCR all pages of an open fitz document."""
    return "\n\n".join(
        _ocr_page(page, ocr_backend, pdf_path) for page in doc
    )


async def _extract_text_async(
    pdf_path: str,
    parser: str = "pymupdf",
    ocr_fallback: bool = False,
    ocr_backend: str = "none",
) -> str:
    return extract_text(
        pdf_path, parser,
        ocr_fallback=ocr_fallback, ocr_backend=ocr_backend,
    )


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


def _make_client_openai(api_key: str | None = None):
    """Build an instructor-wrapped OpenAI async client."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    return instructor.from_openai(AsyncOpenAI(api_key=key))


def _make_client_anthropic(api_key: str | None = None):
    """Build an instructor-wrapped Anthropic async client."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    return instructor.from_anthropic(
        AsyncAnthropic(api_key=key), max_tokens=4096,
    )


def _make_client_litellm(api_key: str | None = None):
    """Build an instructor-wrapped litellm client."""
    from litellm import acompletion
    return instructor.from_litellm(acompletion)


# --- LLM backend registry ---
# Each backend is a callable: (api_key: str | None) -> instructor client

LLM_BACKENDS = {
    "openai": _make_client_openai,
    "anthropic": _make_client_anthropic,
    "litellm": _make_client_litellm,
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

    Returns:
        Populated Pydantic model instance.
    """
    if text is None:
        text = await _extract_text_async(
            pdf_path, parser,
            ocr_fallback=ocr_fallback, ocr_backend=ocr_backend,
        )
    if len(text) > TEXT_WARN_THRESHOLD:
        warnings.warn(
            f"Document is large ({len(text):,} chars). "
            "Results may be incomplete. For tabular schemas, "
            "use extract_pages_async() instead.",
            stacklevel=2,
        )
    client = _make_client(model, api_key, llm_backend)
    return await client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=2,
        messages=_make_messages(text, instructions),
        temperature=0,
        max_tokens=4096,
    )


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
    parser: str = "pymupdf",
    ocr_backend: str = "none",
    llm_backend: str | None = None,
    header_pages: int = 0,
    page_range: str | None = None,
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
            as each chunk completes.
        parser: Text extraction backend — see ``extract_text()``.
        ocr_backend: OCR backend — see ``extract_text()``.
        llm_backend: LLM backend — see ``_make_client()``.
        header_pages: Number of leading pages to treat as a header.
            Their text is prepended to every chunk for context.
        page_range: Optional page range string (e.g. "2-5" or
            "1,3,5-7"). 1-indexed. If omitted, all non-header
            pages are processed.

    Returns:
        List of result dicts (with _page and optionally _error).
    """
    pages = extract_text_pages(pdf_path, parser, ocr_backend=ocr_backend)
    header_text = (
        "\n\n".join(pages[:header_pages]) if header_pages > 0 else ""
    )

    if page_range:
        content_indices = [
            i for i in _parse_page_range(page_range, len(pages))
            if i >= header_pages
        ]
    else:
        content_indices = list(range(header_pages, len(pages)))

    chunks = []
    for chunk_start in range(
        0, len(content_indices), pages_per_chunk
    ):
        idx_slice = content_indices[
            chunk_start: chunk_start + pages_per_chunk
        ]
        text = "\n\n".join(pages[i] for i in idx_slice)
        if header_text:
            text = header_text + "\n\n" + text
        start = idx_slice[0] + 1
        end = idx_slice[-1] + 1
        label = f"p{start}" if start == end else f"p{start}-{end}"
        chunks.append((label, text))

    sem = asyncio.Semaphore(concurrency)
    client = _make_client(model, api_key, llm_backend)
    results = [None] * len(chunks)

    async def _process(idx: int, label: str, text: str):
        async with sem:
            try:
                result = await client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    max_retries=2,
                    messages=_make_messages(text, instructions),
                    temperature=0,
                )
                data = result.model_dump()
                data["_page"] = label
            except Exception as e:
                data = {"_page": label, "_error": str(e)}
            results[idx] = data
            if on_result:
                on_result(label, data)

    await asyncio.gather(
        *[
            _process(i, label, text)
            for i, (label, text) in enumerate(chunks)
        ]
    )
    return results


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

    Returns:
        List of result dicts (with _source_file and optionally
        _error).
    """
    sem = asyncio.Semaphore(concurrency)
    client = _make_client(model, api_key, llm_backend)
    results = []

    async def _process(path: str):
        async with sem:
            try:
                text = await _extract_text_async(
                    path, parser,
                    ocr_fallback=ocr_fallback,
                    ocr_backend=ocr_backend,
                )
                result = await client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    max_retries=2,
                    messages=_make_messages(text, instructions),
                    temperature=0,
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
