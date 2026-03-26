"""EasyOCR plugin for Petey.

Local OCR supporting 80+ languages. Good alternative to Tesseract,
especially for non-English documents.

Requires the ``easyocr`` package: pip install easyocr
"""
import io


def ocr_page(page) -> str:
    """OCR a single fitz page using EasyOCR."""
    try:
        import easyocr
    except ImportError:
        raise ImportError(
            "easyocr is required for ocr_backend='easyocr'. "
            "Install it with: pip install easyocr"
        )
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")

    reader = easyocr.Reader(["en"], verbose=False)
    results = reader.readtext(
        io.BytesIO(img_bytes), detail=0, paragraph=True,
    )
    return "\n".join(results) if results else ""
