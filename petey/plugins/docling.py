"""Docling parser plugin for Petey.

Uses IBM Docling's document converter to extract per-page markdown.
Requires the ``docling`` package: pip install docling
"""


def extract_pages(pdf_path: str) -> list[str]:
    """Extract per-page markdown from a PDF using Docling."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "docling is required for parser='docling'. "
            "Install it with: pip install docling"
        )

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    pages: list[str] = []
    for page_no in range(result.document.num_pages()):
        content_items = [
            item for item in result.document.iterate_items()
            if hasattr(item, "prov")
            and any(p.page_no == page_no + 1 for p in item.prov)
        ]
        md_parts = [
            item.export_to_markdown() for item in content_items
            if hasattr(item, "export_to_markdown")
        ]
        pages.append("\n\n".join(md_parts) if md_parts else "")

    return pages if pages else [""]
