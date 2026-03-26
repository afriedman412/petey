"""Azure Document Intelligence plugin for Petey.

Cloud document parsing via Azure (formerly Form Recognizer).
Strong on forms, tables, and structured documents.

Requires ``azure-ai-documentintelligence``:
    pip install azure-ai-documentintelligence

Requires environment variables:
    AZURE_DOCUMENT_ENDPOINT  — Azure endpoint URL
    AZURE_DOCUMENT_KEY       — Azure API key
"""
import asyncio
import os


async def extract_pages(pdf_path: str) -> list[str]:
    """Extract per-page text using Azure Document Intelligence."""
    try:
        from azure.ai.documentintelligence import (
            DocumentIntelligenceClient,
        )
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        raise ImportError(
            "azure-ai-documentintelligence is required for "
            "parser='azure_documentai'. Install it with: "
            "pip install azure-ai-documentintelligence"
        )

    endpoint = os.environ.get("AZURE_DOCUMENT_ENDPOINT")
    key = os.environ.get("AZURE_DOCUMENT_KEY")
    if not endpoint or not key:
        raise ValueError(
            "Missing config for Azure Document Intelligence "
            "(parser). Set AZURE_DOCUMENT_ENDPOINT and "
            "AZURE_DOCUMENT_KEY in your .env file or environment."
        )

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    loop = asyncio.get_running_loop()

    def _analyze():
        poller = client.begin_analyze_document(
            "prebuilt-read", body=pdf_bytes,
            content_type="application/pdf",
        )
        return poller.result()

    result = await loop.run_in_executor(None, _analyze)

    pages = []
    for page in result.pages:
        lines = [
            line.content for line in (page.lines or [])
            if line.content
        ]
        pages.append("\n".join(lines))

    return pages if pages else [""]
