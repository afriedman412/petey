"""Unstructured API plugin for Petey.

Cloud document parsing via the Unstructured API.

Requires ``unstructured-client``:
    pip install unstructured-client

Requires ``UNSTRUCTURED_API_KEY`` environment variable.
"""

import asyncio
import os


async def extract_pages(pdf_path: str) -> list[str]:
    """Extract per-page text from a PDF using the Unstructured API."""
    try:
        from unstructured_client import UnstructuredClient
        from unstructured_client.models import operations, shared
    except ImportError:
        raise ImportError(
            "unstructured-client is required for parser='unstructured'. "
            "Install it with: pip install unstructured-client"
        )

    api_key = os.environ.get("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise ValueError(
            "UNSTRUCTURED_API_KEY environment variable is required "
            "for parser='unstructured'."
        )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    client = UnstructuredClient(api_key_auth=api_key)

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=pdf_bytes,
                file_name=os.path.basename(pdf_path),
            ),
            split_pdf_allow_failed=True,
        ),
    )

    # unstructured-client is sync — run in a thread
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.general.partition(request=req),
    )

    # Group elements by page number
    page_texts: dict[int, list[str]] = {}
    for element in response.elements:
        page_num = 1
        if hasattr(element, "metadata") and hasattr(element.metadata, "page_number"):
            page_num = element.metadata.page_number or 1
        text = element.get("text", "") if isinstance(element, dict) else getattr(element, "text", "")
        if text:
            page_texts.setdefault(page_num, []).append(text)

    if not page_texts:
        return [""]
    max_page = max(page_texts.keys())
    return [
        "\n\n".join(page_texts.get(p, []))
        for p in range(1, max_page + 1)
    ]
