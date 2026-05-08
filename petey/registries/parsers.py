"""Parser registries — data only.

Built-in parsers (``pymupdf``, ``pdfplumber``) live in ``petey.extract``.
This module declares the two pluggable parser registries:

- ``API_PARSERS``: remote HTTP services that parse a PDF on upload.
- ``PLUGIN_PARSERS``: lazy-imported local parsers in ``petey.plugins``.

Both are mutable dicts — register new entries at runtime by mutating
them (no per-service Python code needed)::

    from petey.registries.parsers import PLUGIN_PARSERS
    PLUGIN_PARSERS["my_parser"] = "my_pkg.pdf:extract_pages"
"""


# --- Remote API backend infrastructure ---
#
# Any HTTP service that accepts a file and returns text can be wired in
# as a parser or OCR backend by adding config to API_PARSERS / API_OCR_BACKENDS.
# No new functions needed — just a dict.
#
# Config keys:
#   endpoint       — URL to POST the file to (required)
#   api_key_env    — env var name for the API key (required)
#   auth_header    — HTTP header name (default: "X-API-Key")
#   auth_prefix    — prepended to key value, e.g. "Bearer" (default: "")
#   request_format — how to send the file (default: "multipart")
#       "multipart"  — standard multipart/form-data file upload
#       "json_b64"   — base64-encoded file in a JSON body
#   file_field     — field name for the file in the request (default: "file")
#   params         — extra form data / JSON fields to include (default: {})
#   response_key   — dot-path into JSON response for the text (default: "markdown")
#   poll           — whether to poll a check URL for async results (default: True)
#   poll_status_key — key to check for completion (default: "status")
#   poll_done_value — value that means done (default: "complete")
#   poll_check_key  — key containing the poll URL (default: "request_check_url")
#   poll_url_template — build poll URL from check_key value via str.format()
#                       e.g. "https://api.example.com/job/{id}/result"
#                       when set, poll_check_key is the value to interpolate
#   poll_header_key — read the poll URL from a response header instead
#                     of the JSON body (e.g. "Operation-Location" for Azure)
#   endpoint_env   — env var containing the base URL (for per-user endpoints)
#   endpoint_suffix — path appended to endpoint_env value
#   timeout        — max seconds to wait for poll (default: 240)
#
# request_format options:
#   "multipart"  — standard multipart/form-data file upload (default)
#   "json_b64"   — base64-encoded file in a JSON body
#   "raw"        — raw file bytes with Content-Type header
#
# response_key patterns:
#   "markdown"    — simple top-level key
#   "result.text" — dot-separated nested key
#   "[].text"     — join text from each element in an array response

API_PARSERS: dict[str, dict] = {
    "datalab": {
        "name": "Datalab",
        "role": "parser",
        "endpoint": "https://www.datalab.to/api/v1/convert",
        "api_key_env": "DATALAB_API_KEY",
        "auth_header": "X-API-Key",
        "params": {"output_format": "markdown"},
        "response_key": "markdown",
        "poll": True,
    },
    "unstructured_api": {
        "name": "Unstructured API",
        "role": "parser",
        "endpoint": "https://api.unstructuredapp.io/general/v0/general",
        "api_key_env": "UNSTRUCTURED_API_KEY",
        "auth_header": "unstructured-api-key",
        "params": {"strategy": "auto"},
        "response_key": "[].text",
        "poll": False,
    },
    "azure_documentai": {
        "name": "Azure Document Intelligence",
        "role": "parser",
        "endpoint_env": "AZURE_DOCUMENT_ENDPOINT",
        "endpoint_suffix": (
            "/documentintelligence"
            "/documentModels/prebuilt-read:analyze"
            "?api-version=2024-11-30"
        ),
        "api_key_env": "AZURE_DOCUMENT_KEY",
        "auth_header": "Ocp-Apim-Subscription-Key",
        "request_format": "raw",
        "response_key": "analyzeResult.content",
        "poll": True,
        "poll_header_key": "Operation-Location",
        "poll_status_key": "status",
        "poll_done_value": "succeeded",
        "timeout": 120,
    },
}


# --- Plugin parsers ---
#
# Register local backends that live outside extract.py. Each entry maps a
# name to a "module.path:callable" string. The callable is lazy-imported
# the first time someone selects that backend, so heavyweight dependencies
# (like docling) are never loaded unless needed.
#
# Callable contract:
#   (pdf_path: str) -> list[str]  (one string per page)
#
# Users can add their own:
#   from petey.registries.parsers import PLUGIN_PARSERS
#   PLUGIN_PARSERS["my_parser"] = "my_package.pdf:extract_pages"

PLUGIN_PARSERS: dict[str, str] = {
    "docling": "petey.plugins.docling:extract_pages",
    "liteparse": "petey.plugins.liteparse:extract_pages",
    "unstructured": "petey.plugins.unstructured:extract_pages",
    "textract": "petey.plugins.textract:extract_pages",
    "google_documentai": "petey.plugins.google_documentai:extract_pages",
}
