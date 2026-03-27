"""LiteParse plugin for Petey.

Fast local PDF parsing with spatial text layout and bounding boxes.
From LlamaIndex. Lightweight — no LLMs, no API keys.

Requires ``liteparse``: pip install liteparse
Also requires the CLI: npm i -g @llamaindex/liteparse
"""


def extract_pages(pdf_path: str) -> list[str]:
    """Extract per-page text from a PDF using LiteParse."""
    try:
        from liteparse import LiteParse
    except ImportError:
        raise ImportError(
            "liteparse is required for parser='liteparse'. "
            "Install it with: pip install liteparse "
            "(also requires: npm i -g @llamaindex/liteparse)"
        )

    parser = LiteParse()
    result = parser.parse(pdf_path)

    pages = []
    for page in result.pages:
        text = " ".join(
            item.text for item in page.textItems
            if hasattr(item, "text") and item.text
        )
        pages.append(text)

    return pages if pages else [""]
