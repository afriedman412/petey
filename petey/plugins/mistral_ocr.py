"""Mistral OCR plugin for Petey.

Uses Mistral's OCR API to extract text from scanned PDF pages.
Requires the ``mistralai`` package: pip install petey[mistral-ocr]
Requires ``MISTRAL_API_KEY`` in the environment.
"""
import base64
import os


def ocr_page(page) -> str:
    """OCR a single fitz page using Mistral's OCR API."""
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
            "Missing API key for Mistral (ocr). "
            "Set MISTRAL_API_KEY in your .env file or environment."
        )
    pix = page.get_pixmap(dpi=200)
    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
    data_url = f"data:image/png;base64,{img_b64}"

    client = Mistral(api_key=api_key)
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "image_url", "image_url": data_url},
    )
    parts = []
    for p in response.pages:
        if p.markdown:
            parts.append(p.markdown)
    return "\n\n".join(parts) if parts else ""
