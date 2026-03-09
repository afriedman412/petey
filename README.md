# Petey

The Easy PDF Extractor. Define a YAML schema, point at your PDFs, get structured data back.

Petey uses LLMs (OpenAI or Anthropic) to extract structured fields from PDF documents. You describe what you want in a YAML schema, and Petey handles text extraction, LLM prompting, and output formatting.

## Install

```bash
pip install .
```

Or in editable/dev mode:

```bash
pip install -e ".[dev]"
```

## Quick start

1. Set your API key:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

2. Write a schema (YAML):

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
    description: Invoice date
  status:
    type: enum
    values: [Paid, Unpaid, Overdue]
    description: Payment status
```

3. Run it:

```bash
petey extract --schema invoice.yaml ./invoices/ -o results.csv
```

## CLI usage

```
petey extract --schema SCHEMA PATHS... [options]
```

| Option | Description |
|---|---|
| `--schema, -s` | YAML schema file (required) |
| `--model, -m` | Model ID (default: `gpt-4.1-mini`, or set `PETEY_MODEL`) |
| `--output, -o` | Output file (`.csv`, `.json`, or `.jsonl`) |
| `--format, -f` | Output format (inferred from `-o` if not set) |
| `--concurrency, -c` | Concurrent API requests (default: 10) |
| `--instructions, -i` | Additional extraction instructions |

`PATHS` can be individual PDF files or directories (all `.pdf` files inside will be processed).

Examples:

```bash
# Single file, JSON to stdout
petey extract -s schema.yaml report.pdf -f json

# Directory, CSV output
petey extract -s schema.yaml ./pdfs/ -o results.csv

# Anthropic model, limited concurrency
petey extract -s schema.yaml ./pdfs/ -m claude-haiku-4-5-20251001 -c 5 -o out.jsonl
```

## Python API

```python
from petey import load_schema, extract, extract_batch

# Load schema
model, spec = load_schema("schema.yaml")

# Single file (sync)
result = extract("doc.pdf", model, model="gpt-4.1-mini")
print(result.model_dump())

# Batch (async)
import asyncio

results = asyncio.run(
    extract_batch(
        ["a.pdf", "b.pdf", "c.pdf"],
        model,
        model="gpt-4.1-mini",
        concurrency=10,
    )
)
```

### Functions

- **`load_schema(path)`** — Load a YAML schema, returns `(PydanticModel, spec_dict)`
- **`build_model(spec)`** — Build a Pydantic model from a spec dict directly
- **`extract(pdf_path, response_model, *, model, api_key, instructions)`** — Extract from one PDF (sync)
- **`extract_async(...)`** — Same as above, async
- **`extract_batch(pdf_paths, response_model, *, model, api_key, instructions, concurrency, on_result)`** — Extract from multiple PDFs concurrently. Optional `on_result(path, data)` callback fires as each file completes.
- **`extract_text(pdf_path)`** — Just get the raw text from a PDF (PyMuPDF)

## Schema format

```yaml
name: MySchema          # optional, used for the Pydantic model name
record_type: array      # optional, use for table extraction (multiple records per doc)
instructions: |         # optional, appended to system prompt
  Focus on the header section for dates.

fields:
  field_name:
    type: string        # string, number, date, enum, or array
    description: What this field contains

  category:
    type: enum
    values: [A, B, C]   # optional — omit to let the LLM infer values

  line_items:
    type: array
    description: Table rows
    fields:
      item: { type: string, description: Item name }
      qty: { type: number, description: Quantity }
```

All fields are nullable — the LLM returns `null` for anything it can't find.

## Development

```bash
make install   # create venv + install with dev deps
make test      # run tests
make clean     # remove venv
```
