"""Surya OCR plugin for Petey.

Local layout-aware OCR from Datalab (same team as Marker).
Good at preserving reading order in complex layouts.

Requires the ``surya-ocr`` package: pip install surya-ocr
"""
import io


def ocr_page(page) -> str:
    """OCR a single fitz page using Surya."""
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
    except ImportError:
        raise ImportError(
            "surya-ocr is required for ocr_backend='surya'. "
            "Install it with: pip install surya-ocr"
        )
    from PIL import Image

    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    predictions = rec_predictor(
        [img], det_predictor=det_predictor,
    )
    if not predictions:
        return ""
    lines = [
        line.text for line in predictions[0].text_lines
        if line.text
    ]
    return "\n".join(lines)
