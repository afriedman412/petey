# Petey

Petey is a framework for PDF data extraction. It wires the PDF parser of your choice to the LLM of your choice, and with a simple schema from the user, pulls structured data out of PDF documents.

```bash
pip install petey
```

For demos, tutorials, and benchmarks, visit [the Petey blog](https://medium.com/@af412).

## Why Petey?

The PDF format was designed to look identical on any screen or printer. It was format and technology agnostic, a universal container for the printed page. But all that mattered was its visual presentation. As long as it rendered correctly, the internal representation didn't matter.

And so the inside of a PDF is often chaotic. It is just a bunch of items — words, characters, shapes, images — and their coordinates, with little or no regard for the relationship between anything. What reads as one cohesive line of text could be three groups of words that happened to be positioned sequentially with the same y-value.

A lot of hard-working folks have developed tools to extract text from PDFs over the years. AI can be a big help too — you don't need a particularly advanced LLM to interpret some fairly difficult documents. But models need infrastructure, and not everyone has time to wire it all together.

Petey does the wiring for you. Just pass it your files and a schema that explains what you want, and it returns a JSON or CSV with your data.

## How it works

1. **Parse** — extract text from the PDF using a local or cloud parser
2. **LLM** — send the text to an LLM with your schema to get structured fields back
3. **Output** — return the results as JSON or CSV

## Parsers

| Parser | Install | Best for |
|--------|---------|----------|
| `pymupdf` | included | Most documents. Reads embedded text directly. Fast, free, default. |
| `pdfplumber` | included | Borderless tables. Layout-preserving spatial extraction. |
| `tables` | included | Bordered tables. Uses PyMuPDF's table detection. |
| `marker` | included | Complex/scanned layouts. Remote API via Datalab. Requires `DATALAB_API_KEY`. |
| `unstructured` | included | General-purpose. Remote API. Requires `UNSTRUCTURED_API_KEY`. |

See `petey list parsers` for all available parsers.

## OCR Backends

If a PDF has no embedded text (e.g. scanned documents), Petey falls back to OCR. Only triggered when extracted text is very short.

| Backend | Install | How it works |
|---------|---------|--------------|
| `none` | — | No OCR. Default. |
| `tesseract` | included | Local OCR. Requires Tesseract binary installed. |
| `chandra` | included | Cloud OCR via Datalab. Requires `DATALAB_API_KEY`. |
| `surya` | included | Cloud OCR via Datalab. Requires `DATALAB_API_KEY`. |
| `mistral` | `pip install petey[mistral-ocr]` | Cloud OCR via Mistral. Requires `MISTRAL_API_KEY`. |

See `petey list ocr` for all available backends.

## LLM Backends

Petey auto-detects the right backend from the model name.

| Backend | Models | Auto-detected when |
|---------|--------|--------------------|
| `openai` | `gpt-4.1-mini`, `gpt-4o`, etc. | Default |
| `anthropic` | `claude-sonnet-4-6`, `claude-haiku-4-5`, etc. | Model starts with `claude` |
| `litellm` | Gemini, DeepSeek, Fireworks, Ollama, Bedrock, 100+ more | Model has a provider prefix (e.g. `gemini/`, `deepseek/`, `fireworks_ai/`) |

## Setup

Add your API key to a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Or for other providers:

```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DATALAB_API_KEY=...
```

## Schemas

Every extraction starts with a schema — a YAML file that tells Petey what to look for.

```yaml
name: Invoice
fields:
  vendor:
    type: string
    description: Company name on the invoice
  amount:
    type: number
    description: Total amount due
  date:
    type: date
  status:
    type: category
    values: [Paid, Unpaid, Overdue]
```

### Field types

| Type | Notes |
|------|-------|
| `string` | Any text value |
| `number` | Integer or decimal |
| `date` | Returns ISO 8601 format |
| `category` | Constrained set of values. List `values:` to enforce them. Case-insensitive matching. |

All fields are nullable — Petey returns `null` for anything it can't find rather than guessing.

### Schema options

| Option | Description |
|--------|-------------|
| `mode: table` | Extract multiple records per page (default: `query` — one record per file) |
| `instructions` | Extra guidance appended to the prompt |
| `header_pages` | Number of leading pages to prepend to every chunk (for context like column headers) |
| `pages` | Page range to process, e.g. `"2-5"` or `"1,3,5-7"` |
| `input` | Default PDF path or directory |
| `output` | Default output file path |
| `parser` | Default parser |
| `ocr` | Default OCR backend |

## CLI

```bash
# Basic extraction
petey extract --schema invoice.yaml ./invoices/ -o results.csv

# With options
petey extract --schema schema.yaml --model claude-sonnet-4-6 --parser marker ./pdfs/

# List available backends
petey list parsers
petey list ocr
petey list llm
```

| Flag | Default | Description |
|------|---------|-------------|
| `--schema / -s` | required | Path to YAML schema |
| `--model / -m` | `gpt-4.1-mini` | LLM model ID |
| `--parser` | `pymupdf` | Text extraction backend |
| `--ocr` | `none` | OCR backend |
| `--concurrency / -c` | `10` | Max concurrent API calls |
| `--output / -o` | stdout | Output file path |
| `--format / -f` | inferred | `csv`, `json`, or `jsonl` |
| `--mode` | from schema | `query` or `table` |
| `--header-pages` | from schema | Header pages to prepend to each chunk |
| `--page-range` | from schema | Page range to extract |

## Python API

```python
from petey import extract, load_schema

schema, spec = load_schema("invoice.yaml")

result = extract("invoice.pdf", schema)

# With options
result = extract(
    "invoice.pdf",
    schema,
    model="claude-sonnet-4-6",
    parser="marker",
    ocr_backend="chandra",
)
```

## Optional Dependencies

```bash
pip install petey                    # Core (pymupdf, pdfplumber, tesseract, litellm)
pip install petey[mistral-ocr]       # + Mistral OCR
pip install petey[tabula]            # + Tabula table extraction (requires Java)
pip install petey[unstructured]      # + Unstructured API client
pip install petey[all]               # Everything
```
