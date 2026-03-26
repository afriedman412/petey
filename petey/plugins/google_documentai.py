"""Google Document AI plugin for Petey.

Cloud document parsing via Google Cloud. Strong on structured
documents, forms, and tables.

Requires ``google-cloud-documentai``:
    pip install google-cloud-documentai

Requires environment variables:
    GCP_PROJECT_ID       — Google Cloud project ID
    GCP_LOCATION         — processor location (default: us)
    GCP_PROCESSOR_ID     — Document AI processor ID
"""
import asyncio
import os


async def extract_pages(pdf_path: str) -> list[str]:
    """Extract per-page text from a PDF using Google Document AI."""
    try:
        from google.cloud import documentai_v1 as documentai
    except ImportError:
        raise ImportError(
            "google-cloud-documentai is required for "
            "parser='google_documentai'. Install it with: "
            "pip install google-cloud-documentai"
        )

    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_LOCATION", "us")
    processor_id = os.environ.get("GCP_PROCESSOR_ID")
    if not project_id or not processor_id:
        raise ValueError(
            "Missing config for Google Document AI (parser). "
            "Set GCP_PROJECT_ID and GCP_PROCESSOR_ID in your "
            ".env file or environment."
        )

    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    with open(pdf_path, "rb") as f:
        content = f.read()

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(
            content=content,
            mime_type="application/pdf",
        ),
    )

    # SDK is sync — run in a thread
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: client.process_document(request=request),
    )
    document = result.document

    pages = []
    for page in document.pages:
        lines = []
        for line in page.lines:
            text_anchor = line.layout.text_anchor
            segments = []
            for seg in text_anchor.text_segments:
                start = int(seg.start_index) if seg.start_index else 0
                end = int(seg.end_index)
                segments.append(document.text[start:end])
            lines.append("".join(segments))
        pages.append("\n".join(lines))

    return pages if pages else [""]
