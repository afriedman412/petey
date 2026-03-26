"""Google Cloud Vision OCR plugin for Petey.

Cloud OCR via Google Cloud Vision API.

Requires ``google-cloud-vision``:
    pip install google-cloud-vision

Requires Google Cloud credentials configured
(GOOGLE_APPLICATION_CREDENTIALS env var or default credentials).
"""
import asyncio


async def ocr_page(page) -> str:
    """OCR a single fitz page using Google Cloud Vision."""
    try:
        from google.cloud import vision
    except ImportError:
        raise ImportError(
            "google-cloud-vision is required for "
            "ocr_backend='google_vision'. Install it with: "
            "pip install google-cloud-vision"
        )

    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=img_bytes)

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.document_text_detection(image=image),
    )

    if response.full_text_annotation:
        return response.full_text_annotation.text
    return ""
