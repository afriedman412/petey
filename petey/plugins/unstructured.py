"""Unstructured plugin for Petey.

Local document parsing via the unstructured library.
Handles PDFs, images, HTML, and more.

Requires ``unstructured``:
    pip install unstructured[pdf]
"""


def extract_pages(pdf_path: str) -> list[str]:
    """Extract per-page text from a PDF using unstructured."""
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        raise ImportError(
            "unstructured is required for parser='unstructured'. "
            "Install it with: pip install unstructured[pdf]"
        )

    elements = partition_pdf(filename=pdf_path)

    page_texts: dict[int, list[str]] = {}
    for el in elements:
        page_num = el.metadata.page_number or 1
        if el.text:
            page_texts.setdefault(page_num, []).append(el.text)

    if not page_texts:
        return [""]
    max_page = max(page_texts.keys())
    return [
        "\n\n".join(page_texts.get(p, []))
        for p in range(1, max_page + 1)
    ]
