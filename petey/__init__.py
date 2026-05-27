"""Petey — The Easy PDF Extractor."""

from petey.schema import (
    build_model,
    load_blueprint,
    load_schema,  # deprecated; will be removed in v0.6.0
    normalize_dates,
)
from petey.extract import (
    extract,
    extract_async,
    extract_batch,
    extract_text,
    extract_text_pages,
    extract_pages_async,
    infer_blueprint,
    infer_blueprint_async,
    infer_blueprint_vision_async,
    # Deprecated; will be removed in v0.6.0
    infer_schema,
    infer_schema_async,
    infer_schema_vision_async,
)

__all__ = [
    "build_model",
    "load_blueprint",
    "load_schema",
    "normalize_dates",
    "extract",
    "extract_async",
    "extract_batch",
    "extract_text",
    "extract_text_pages",
    "extract_pages_async",
    "infer_blueprint",
    "infer_blueprint_async",
    "infer_blueprint_vision_async",
    "infer_schema",
    "infer_schema_async",
    "infer_schema_vision_async",
]
