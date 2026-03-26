# Petey

Meet Petey. Petey gets your data out of PDFs and into formats that you can actually work with. It formats weird tables into easy-to-analyze CSV documents. It can even answer questions about what's in your data. It's free to use, beyond a negligible hit to your OpenAI or Anthropic account.

```bash
pip install petey
```

## Why Petey?

The PDF format was designed to look identical on any screen or printer. It was format and technology agnostic, a universal container for the printed page. But all that mattered was its visual presentation. As long as it rendered correctly, the internal representation didn't matter.

And so the inside of a PDF is often chaotic. It is just a bunch of items — words, characters, shapes, images — and their coordinates, with little or no regard for the relationship between anything. What reads as one cohesive line of text could be three groups of words that happened to be positioned sequentially with the same y-value. You have probably experienced this if you have ever tried to select text in a PDF and find yourself highlighting disconnected words on four different lines.

The primary challenge of getting your data out of a PDF is imposing some order on this code soup. A lot of hard-working folks have developed methods and tools to accomplish this over the years, finding ingenious ways to infer meaning from the relative positions of all the elements of a document. But it's tricky, precise work.

As it turns out, AI can be a big help here. It turns out you don't need a particularly advanced LLM to interpret some fairly difficult PDFs. But models need infrastructure. Not everyone has time to teach themselves how to be an amateur ML engineer, even with the magic of vibe coding.

Petey wires everything together for you. Just pass it your files and a schema that explains what you want, and it will return a JSON or CSV with your data.

## Architecture
### Text Extractors

The first step is getting text out of the PDF. Different extractors work better for different document types.

| Parser | Install | Best for | How it works |
|--------|---------|----------|--------------|
| `pymupdf` | included | Most documents | Reads the embedded text layer directly. Fast and reliable. Default for single-record schemas. |
| `tables` | included | Bordered tables | Uses PyMuPDF's table detection to find cells and extract them as TSV. Falls back to plain text for pages without tables. Default for array schemas. |
| `pdfplumber` | included | Borderless tables | Layout-preserving extraction that positions text spatially. Good for documents where columns are aligned by whitespace rather than cell borders. |
| `tabula` | `pip install petey[tabula]` | Structured tables | Uses tabula-py (Java-based) to detect and extract tables as DataFrames. Falls back to pymupdf for pages without tables. Requires Java. |
| `marker` | included | Complex layouts | Remote API via Datalab. Requires `DATALAB_API_KEY`. |
| `llamaparse` | included | Complex layouts | Remote API via LlamaCloud. Requires `LLAMA_CLOUD_API_KEY`. |

Parsers are registered in the `PARSERS` dict. Local parsers are sync functions; API parsers (like `marker`) are async and automatically routed through the API concurrency pool. To add a new API parser, add an entry to `API_PARSERS` — no new code needed.

### OCR Backends

If a PDF has no embedded text layer (e.g. a scanned document), Petey can fall back to OCR. OCR is only triggered when the extracted text is very short (< 100 characters), so it won't slow down documents that already have text.

| Backend | Install | How it works |
|---------|---------|--------------|
| `none` | — | No OCR. Default. |
| `tesseract` | included | Local OCR via pytesseract. Requires the Tesseract binary installed on your system. |
| `mistral` | `pip install petey[mistral-ocr]` | Cloud OCR via Mistral's `mistral-ocr-latest` model. Sends page images to the Mistral API. Requires `MISTRAL_API_KEY`. |
| `chandra` | included | Cloud OCR via Datalab. Requires `DATALAB_API_KEY`. |

Like parsers, OCR backends are registered in `OCR_BACKENDS`. API backends are async and can be added via `API_OCR_BACKENDS`.

### LLM Backends

The LLM interprets the extracted text and returns structured data matching your schema. Petey auto-detects the right backend from the model name, or you can set it explicitly.

| Backend | Install | Models | Auto-detected when |
|---------|---------|--------|--------------------|
| `openai` | included | `gpt-4.1-mini`, `gpt-4o`, etc. | Model name doesn't match other patterns (default) |
| `anthropic` | included | `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`, etc. | Model starts with `claude` |
| `litellm` | included | Gemini, Mistral, Ollama, Bedrock, Vertex AI, Cohere, 100+ more | Model starts with `gemini/`, `mistral/`, `ollama/`, `bedrock/`, etc. |

### Adding Custom Backends

Each pipeline step has a config dict for adding API-based backends without writing code:

```python
from petey.extract import API_PARSERS, API_OCR_BACKENDS, API_LLM_BACKENDS

# Add a new PDF parser
API_PARSERS["my_parser"] = {
    "endpoint": "https://my-api.com/parse",
    "api_key_env": "MY_API_KEY",
    "response_key": "markdown",
    "poll": True,  # poll for async results
}

# Add a new OCR backend
API_OCR_BACKENDS["my_ocr"] = {
    "endpoint": "https://my-api.com/ocr",
    "api_key_env": "MY_API_KEY",
    "response_key": "text",
    "poll": False,  # synchronous response
}

# Add an OpenAI-compatible LLM endpoint
API_LLM_BACKENDS["my_llm"] = {
    "client": "openai",
    "base_url": "https://my-host.com/v1",
    "api_key_env": "MY_LLM_KEY",
}
```

API parser and OCR backends are automatically async and route through the API concurrency pool.

### Concurrency

Petey uses a process-wide `ConcurrencyManager` with two pools:

- **CPU pool** — local parsing (pymupdf, pdfplumber, tesseract). Backed by a `ProcessPoolExecutor`, sized to CPU cores.
- **API pool** — remote calls (LLM, Marker, Chandra, Mistral OCR). Bounded by an `asyncio.Semaphore`, configurable via `--concurrency` (default 10).

The manager automatically dispatches work to the right pool based on whether the callable is sync (CPU) or async (API). For array schemas, pages are subsetted into individual temp PDFs and dispatched independently, so parsing page N and LLM extraction of page N-1 happen in parallel.

```python
from petey.concurrency import configure

# Adjust limits
configure(cpu_limit=4, api_limit=20)
```

## Setup

Add your API key to a `.env` file in your working directory:

```
OPENAI_API_KEY=sk-...
```

Or for Anthropic:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Or for any provider via litellm (e.g. Gemini):

```
GEMINI_API_KEY=...
```

Petey defaults to `gpt-4.1-mini`, which is fast and cheap and handles most documents well. To use a different model, pass `--model` on the CLI or set a default in `.env`:

```
PETEY_MODEL=claude-sonnet-4-6
```

## Schemas

Every extraction starts with a schema — a simple YAML file that tells Petey what fields to look for and what type of data to expect in each one.

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
| `category` | Constrained set of values; optionally list `values:` to enforce them |

All fields are nullable — Petey returns `null` for anything it can't find rather than guessing.

### Schema options

| Option | Description |
|--------|-------------|
| `input` | Path to a PDF file or directory of PDFs. Overridden by CLI positional args. |
| `output` | Output file path. Format inferred from extension (`.csv`, `.json`, `.jsonl`). Overridden by `-o`. Defaults to CSV for table mode, JSON for query mode. |
| `mode: table` | Extract multiple records per page (default: `query` — one record per file). Accepts `record_type: array` for backwards compatibility. |
| `instructions` | Extra guidance appended to the prompt (e.g. "ignore the summary row") |
| `header_pages` | Number of leading pages to treat as a document header (see below) |
| `pages` | Page range to process, e.g. `"2-5"` or `"1,3,5-7"` (1-indexed) |
| `parser` | Text extraction backend: `pymupdf` (default), `tables`, `pdfplumber`, `tabula`, `marker`, or `llamaparse` |

## Use Cases

### One record per file

The simplest case: you have a pile of documents that all contain the same kind of information, and you want one row of data out of each one. Think invoices, contracts, permit applications, or any form where each file is a self-contained record. This works on narrative text too — if you have 200 complaint letters and want to extract the filer name, date, and subject from each one, Petey handles that just as well as a structured form.

Petey processes each file and returns a single structured record. Run it against a folder and you get a CSV with one row per file.

```yaml
name: Invoice
fields:
  vendor:
    type: string
    description: Name of the vendor
  invoice_number:
    type: string
  date:
    type: date
  amount_due:
    type: number
  status:
    type: category
    values: [Paid, Unpaid, Overdue]
```

```bash
petey extract --schema invoice.yaml ./invoices/ -o results.csv
```

### One table per file

Some documents contain a table of records — a bank statement, a schedule of assets, a list of transactions. Add `mode: table` and Petey will return multiple rows per file. Petey splits the document into pages, processes each concurrently, and assembles the results in document order.

Documents like this often have important context on the first page — a header, column labels, the filer's name. The `header_pages` option tells Petey to prepend those pages to every page it sends to the LLM, so that context is always visible no matter how deep into the document it's looking.

```yaml
name: Transactions
mode: table
header_pages: 1
fields:
  date:
    type: date
  description:
    type: string
  amount:
    type: number
```

```bash
petey extract --schema transactions.yaml statement.pdf -o results.csv
```

You can point this at a folder of files and Petey will combine all the rows into a single output, with a `_source_file` column on each row. Use `pages` to target a specific range if the table only occupies part of the document (e.g. `pages: "3-22"`).

### Scanned documents

If your PDFs are scanned images with no text layer, use an OCR backend:

```bash
# Local OCR with Tesseract
petey extract --schema schema.yaml --ocr tesseract ./scans/ -o results.csv

# Cloud OCR with Mistral (better quality, requires API key)
petey extract --schema schema.yaml --ocr mistral ./scans/ -o results.csv
```

### Using different LLM providers

```bash
# OpenAI (default)
petey extract --schema schema.yaml --model gpt-4.1-mini ./pdfs/

# Anthropic
petey extract --schema schema.yaml --model claude-sonnet-4-6 ./pdfs/

# Gemini via litellm
petey extract --schema schema.yaml --model gemini/gemini-2.0-flash ./pdfs/

# Ollama (local)
petey extract --schema schema.yaml --model ollama/llama3 ./pdfs/

# Explicit backend override
petey extract --schema schema.yaml --model my-model --llm-backend litellm ./pdfs/
```

## CLI Reference

```bash
petey extract --schema schema.yaml ./pdfs/ -o results.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--schema / -s` | required | Path to your YAML schema |
| `--model / -m` | `gpt-4.1-mini` | Model ID to use |
| `--concurrency / -c` | `10` | Number of concurrent API calls |
| `--output / -o` | stdout | Output file path |
| `--format / -f` | inferred | `csv`, `json`, or `jsonl` |
| `--instructions / -i` | — | Extra extraction instructions |
| `--parser` | `pymupdf` | Text extraction backend (`pymupdf`, `tables`, `pdfplumber`, `tabula`, `marker`, `llamaparse`) |
| `--ocr` | `none` | OCR backend (`none`, `tesseract`, `mistral`, `chandra`) |
| `--llm-backend / -b` | auto-detect | LLM backend (`openai`, `anthropic`, `litellm`) |
| `--pages-per-chunk / -p` | `1` for arrays | Pages per LLM call (set to `0` to disable chunking) |

## Python API

```python
from petey import extract, load_schema

# load_schema returns a Pydantic model class and the raw spec dict
schema, spec = load_schema("invoice.yaml")

# Basic extraction
result = extract("invoice.pdf", schema)

# With options
result = extract(
    "invoice.pdf",
    schema,
    model="claude-sonnet-4-6",      # LLM model ID
    parser="pdfplumber",            # text extraction backend
    ocr_backend="mistral",          # OCR for scanned docs
    llm_backend="litellm",          # explicit LLM backend
    instructions="Ignore the watermark text",
)
```

## Optional Dependencies

```bash
pip install petey                    # Core (includes tesseract OCR + litellm)
pip install petey[mistral-ocr]       # + Mistral OCR
pip install petey[tabula]            # + Tabula table extraction (requires Java)
```