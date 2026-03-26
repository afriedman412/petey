"""PaddleOCR plugin for Petey.

Local OCR from Baidu. Strongest option for CJK languages,
also very good for English.

Requires the ``paddleocr`` and ``paddlepaddle`` packages:
    pip install paddleocr paddlepaddle
"""
import io
import numpy as np


def ocr_page(page) -> str:
    """OCR a single fitz page using PaddleOCR."""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError(
            "paddleocr is required for ocr_backend='paddleocr'. "
            "Install it with: pip install paddleocr paddlepaddle"
        )
    from PIL import Image

    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    img_array = np.array(img)

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    results = ocr.ocr(img_array, cls=True)

    if not results or not results[0]:
        return ""
    lines = [
        line[1][0] for line in results[0]
        if line[1] and line[1][0]
    ]
    return "\n".join(lines)
