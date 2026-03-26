"""Amazon Textract plugin for Petey.

Cloud document extraction via AWS. Best-in-class for forms and
tables with key-value pair detection.

Requires ``boto3``: pip install boto3
Requires AWS credentials configured (env vars, ~/.aws/credentials,
or IAM role).
"""
import asyncio


async def extract_pages(pdf_path: str) -> list[str]:
    """Extract per-page text from a PDF using Amazon Textract."""
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for parser='textract'. "
            "Install it with: pip install boto3"
        )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    client = boto3.client("textract")

    # boto3 is sync — run in a thread to avoid blocking the loop
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.detect_document_text(
            Document={"Bytes": pdf_bytes},
        ),
    )

    # Group blocks by page
    page_texts: dict[int, list[str]] = {}
    for block in response.get("Blocks", []):
        if block["BlockType"] == "LINE":
            page_num = block.get("Page", 1)
            page_texts.setdefault(page_num, []).append(
                block.get("Text", "")
            )

    if not page_texts:
        return [""]
    max_page = max(page_texts.keys())
    return [
        "\n".join(page_texts.get(p, []))
        for p in range(1, max_page + 1)
    ]
