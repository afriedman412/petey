"""
Core PDF extraction. No web dependencies.
Configure via environment variables or pass explicitly.
"""
import asyncio
import os

import fitz
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


def extract_text(
    pdf_path: str,
    parser: str = "pymupdf",
    ocr_fallback: bool = False,
) -> str:
    if parser == "docparse":
        return _extract_text_docparse(pdf_path)
    doc = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text("text") for page in doc)
    if ocr_fallback and len(text.strip()) < OCR_THRESHOLD:
        return _ocr_pymupdf(doc)
    return text


def extract_text_pages(
    pdf_path: str,
    parser: str = "pymupdf",
    ocr_fallback: bool = False,
) -> list[str]:
    """Extract text from each page of a PDF separately."""
    if parser == "docparse":
        return [_extract_text_docparse(pdf_path)]
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if ocr_fallback and len(text.strip()) < OCR_THRESHOLD:
            text = _ocr_page(page)
        pages.append(text)
    return pages


def _ocr_pymupdf(doc) -> str:
    """OCR all pages of an open fitz document using pytesseract."""
    return "\n\n".join(_ocr_page(page) for page in doc)


def _ocr_page(page) -> str:
    """OCR a single fitz page using pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required for OCR fallback. "
            "Install them with: pip install petey[ocr]"
        )
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _extract_text_docparse(pdf_path: str, api_key: str | None = None) -> str:
    try:
        from aryn_sdk.partition import partition_file
    except ImportError:
        raise ImportError(
            "aryn-sdk is required for parser='docparse'. "
            "Install it with: pip install aryn-sdk"
        )
    key = api_key or os.environ.get("DOCPARSE_API_KEY") or os.environ.get("ARYN_API_KEY")
    kwargs = {"aryn_api_key": key} if key else {}
    with open(pdf_path, "rb") as f:
        result = partition_file(f, **kwargs)
    texts = [
        el.get("text_representation", "")
        for el in result.get("elements", [])
        if el.get("text_representation")
    ]
    return "\n\n".join(texts)


async def _extract_text_async(
    pdf_path: str,
    parser: str = "pymupdf",
    docparse_api_key: str | None = None,
    ocr_fallback: bool = False,
) -> str:
    """Async text extraction — runs DocParse in a thread to avoid blocking."""
    if parser == "docparse":
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: _extract_text_docparse(pdf_path, docparse_api_key)
        )
    return extract_text(pdf_path, parser, ocr_fallback=ocr_fallback)


# --- LLM client ---

def _get_provider(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"


def _make_client(model: str, api_key: str | None = None):
    """Build an instructor-wrapped async client for the given model."""
    provider = _get_provider(model)
    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        return instructor.from_anthropic(AsyncAnthropic(api_key=key))
    else:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        return instructor.from_openai(AsyncOpenAI(api_key=key))


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
    docparse_api_key: str | None = None,
    ocr_fallback: bool = False,
    text: str | None = None,
) -> BaseModel:
    """Extract structured data from a PDF."""
    if text is None:
        text = await _extract_text_async(
            pdf_path, parser, docparse_api_key, ocr_fallback=ocr_fallback
        )
    client = _make_client(model, api_key)
    return await client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=2,
        messages=_make_messages(text, instructions),
        temperature=0,
    )


def extract(
    pdf_path: str,
    response_model: type[BaseModel],
    *,
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
    instructions: str = "",
    parser: str = "pymupdf",
    docparse_api_key: str | None = None,
    ocr_fallback: bool = False,
) -> BaseModel:
    """Sync wrapper around extract_async."""
    return asyncio.run(
        extract_async(
            pdf_path, response_model,
            model=model, api_key=api_key, instructions=instructions,
            parser=parser, docparse_api_key=docparse_api_key,
            ocr_fallback=ocr_fallback,
        )
    )


# --- Page-chunked extraction ---

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
    docparse_api_key: str | None = None,
) -> list[dict]:
    """
    Split a PDF into page chunks and extract each concurrently.

    Args:
        pages_per_chunk: Number of pages per chunk (default 1).
        concurrency: Max concurrent API calls.
        on_result: Optional callback(chunk_label, data_dict) called as each
            chunk completes.

    Returns list of result dicts (with _page and optionally _error).
    """
    pages = extract_text_pages(pdf_path, parser)
    chunks = []
    for i in range(0, len(pages), pages_per_chunk):
        chunk_pages = pages[i: i + pages_per_chunk]
        text = "\n\n".join(chunk_pages)
        start = i + 1
        end = min(i + pages_per_chunk, len(pages))
        label = f"p{start}" if start == end else f"p{start}-{end}"
        chunks.append((label, text))

    sem = asyncio.Semaphore(concurrency)
    client = _make_client(model, api_key)
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
        *[_process(i, label, text) for i, (label, text) in enumerate(chunks)]
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
    docparse_api_key: str | None = None,
    ocr_fallback: bool = False,
) -> list[dict]:
    """
    Extract from multiple PDFs concurrently.

    Args:
        on_result: Optional callback(path, data_dict) called as each file completes.
        concurrency: Max concurrent API calls.
        parser: PDF text extraction backend ('pymupdf' or 'docparse').
        docparse_api_key: DocParse API key (or set DOCPARSE_API_KEY env var).
        ocr_fallback: Fall back to pytesseract if pymupdf finds no text layer.

    Returns list of result dicts (with _source_file and optionally _error).
    """
    sem = asyncio.Semaphore(concurrency)
    client = _make_client(model, api_key)
    results = []

    async def _process(path: str):
        async with sem:
            try:
                text = await _extract_text_async(
                    path, parser, docparse_api_key, ocr_fallback=ocr_fallback
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
