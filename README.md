# Petey

Petey is a framework for PDF data extraction. It wires the PDF parser of your choice to the LLM of your choice, and with a simple blueprint from the user, pulls data out of PDF documents.

```bash
pip install petey
```

For the web version, demos and tutorials, visit [Petey](https://petey.cc/demos).

## Why Petey?

The PDF format was designed to look identical on any screen or printer. It was format and technology agnostic, a universal container for the printed page. But all that mattered was its visual presentation. As long as it rendered correctly, the internal representation didn't matter.

And so the inside of a PDF is often chaotic. It is just a bunch of items — words, characters, shapes, images — and their coordinates, with little or no regard for the relationship between anything. What reads as one cohesive line of text could be three groups of words that happened to be positioned sequentially with the same y-value.

A lot of hard-working folks have developed tools to extract text from PDFs over the years. AI can be a big help too — you don't need a particularly advanced LLM to interpret some fairly difficult documents. But models need infrastructure, and not everyone has time to wire it all together.

Petey does the wiring for you. Just pass it your files and a blueprint that explains what you want, and it returns a JSON or CSV with your data.

## How it works

1. **Parse** — extract text from the PDF using a local or cloud parser
2. **LLM** — send the text to an LLM with your blueprint to get the fields you want back
3. **Output** — return the results as JSON or CSV

## Parsers

| Parser | Install | Best for |
|--------|---------|----------|
| `pymupdf` | included | Most documents. Reads embedded text directly, auto-OCRs scanned pages. Fast, free, default. |
| `pdfplumber` | included | Borderless tables. Layout-preserving spatial extraction. Text-only (no OCR). |
| `datalab` | included | Scanned/complex layouts. Remote API via Datalab. Requires `DATALAB_API_KEY`. |
| `unstructured` | included | General-purpose. Remote API. Requires `UNSTRUCTURED_API_KEY`. |

See `petey list parsers` for all available parsers.

## LLM Backends

Petey ships direct, hand-coded backends per provider family and uses `litellm` only as a fallback for the long tail. The right backend is auto-detected from the model name; override with `--llm-backend` when the default isn't right (e.g. running `gpt-4o` through Azure rather than direct OpenAI).

| Backend | Models | Auto-detected when |
|---------|--------|--------------------|
| `openai` | `gpt-4.1-mini`, `gpt-4o`, etc. | Default; model starts with `gpt-`, `o1`, `o3`, `o4` |
| `anthropic` | `claude-sonnet-4-6`, `claude-haiku-4-5`, etc. | Model starts with `claude` |
| `azure_openai` | Any OpenAI deployment on Azure | Pass `--llm-backend azure_openai` |
| `ollama` | Local models via Ollama's OpenAI-compat endpoint | Model starts with `ollama/` |
| `gemini` | `gemini-2.5-flash`, etc. (direct, via `google-genai`) | Model starts with `gemini/` |
| `anthropic_bedrock` | Claude on AWS Bedrock | Pass `--llm-backend anthropic_bedrock` |
| `anthropic_vertex` | Claude on GCP Vertex | Pass `--llm-backend anthropic_vertex` |
| `vertex_ai` | Gemini/Gemma/Llama on GCP Vertex | Pass `--llm-backend vertex_ai` |
| OpenAI-compat catchalls | DeepSeek, Mistral, Together, OpenRouter, Fireworks, Groq | Model has the provider prefix (e.g. `deepseek/`, `mistral/`) |
| `litellm` | Bedrock, Cohere, Replicate, HuggingFace, … | Long-tail prefixes only |

Run `petey list llm` to see every backend wired up in your install.

### Custom model registry

The built-in registry covers common cases. To add your own — e.g. an Azure OpenAI tenant with its own endpoint, or a remote Ollama host — edit `~/.petey/models.yaml`:

```bash
petey models init      # writes a commented template
petey models path      # prints the resolved file path
petey models list      # shows all registered models with provenance
```

```yaml
# ~/.petey/models.yaml
my-azure-gpt-4o:
  provider: azure_openai
  model: gpt-4o                                            # Azure deployment name
  config:
    api_version: "2024-06-01"
    azure_endpoint: https://my-tenant.openai.azure.com
    api_key_env: MY_AZURE_KEY                              # env var holding the key

remote-qwen:
  provider: ollama
  model: qwen2.5:7b
  config:
    base_url: http://gpu-box.local:11434/v1
```

Then `petey extract -m my-azure-gpt-4o ...` works from any directory. User-config entries override built-ins on key collision. Use `$PETEY_MODELS=path/to/file.yaml` to point at a different file, or `--models-config PATH` for a one-off run.

## Setup

Add your API key to a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Or for other providers:

```
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
MISTRAL_API_KEY=...
TOGETHER_API_KEY=...
OPENROUTER_API_KEY=...
FIREWORKS_API_KEY=...
GROQ_API_KEY=...
DATALAB_API_KEY=...
```

Azure OpenAI, Bedrock, and Vertex use platform-specific auth (`OPENAI_API_BASE` + `API_VERSION`, AWS boto3 chain, GCP service account). For those, register the deployment in `~/.petey/models.yaml` (see [Custom model registry](#custom-model-registry)) so endpoint and version travel with the model name.

## Blueprints

Every extraction starts with a blueprint — a `.bpt` file (YAML format) that tells Petey what to look for.

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

### Blueprint options

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
petey extract --blueprint invoice.bpt ./invoices/ -o results.csv

# With options
petey extract --blueprint blueprint.bpt --model claude-sonnet-4-6 --parser datalab ./pdfs/

# Route a model through a non-default backend (here: gpt-4o on Azure)
petey extract --blueprint blueprint.bpt -m gpt-4o --llm-backend azure_openai ./pdfs/

# Inspect what's available
petey list parsers
petey list llm
petey models list
```

| Flag | Default | Description |
|------|---------|-------------|
| `--blueprint / -b` | required | Path to blueprint file (`.bpt` or `.yaml`) |
| `--model / -m` | `gpt-4.1-mini` | LLM model ID |
| `--llm-backend` | from registry | Override the LLM backend (e.g. `azure_openai`); reads its config from env vars |
| `--models-config` | none | Per-run YAML of model registry entries (in addition to `~/.petey/models.yaml`) |
| `--parser` | `pymupdf` | Text extraction backend |
| `--concurrency / -c` | `10` | Max concurrent API calls |
| `--output / -o` | stdout | Output file path |
| `--format / -f` | inferred | `csv`, `json`, or `jsonl` |
| `--mode` | from blueprint | `query` or `table` |
| `--header-pages` | from blueprint | Header pages to prepend to each chunk |
| `--page-range` | from blueprint | Page range to extract |

## Python API

```python
from petey import extract, load_blueprint

response_model, spec = load_blueprint("invoice.bpt")

result = extract("invoice.pdf", response_model)

# With options
result = extract(
    "invoice.pdf",
    response_model,
    model="claude-sonnet-4-6",
    parser="datalab",
    llm_backend="azure_openai",   # optional override
)
```

Custom models registered in `~/.petey/models.yaml` are picked up automatically — no code changes needed; just reference the entry by name in `model=`.

## Migrating from v0.4.x → v0.5.0

User-facing concepts have been renamed from "schema" to "blueprint" and the file extension from `.yaml` to `.bpt`. The YAML format itself is unchanged.

| v0.4.x | v0.5.0 |
|---|---|
| `load_schema(...)` | `load_blueprint(...)` |
| `infer_schema(...)` / `infer_schema_async(...)` / `infer_schema_vision_async(...)` | `infer_blueprint(...)` / `infer_blueprint_async(...)` / `infer_blueprint_vision_async(...)` |
| `petey extract --schema my.yaml ...` | `petey extract --blueprint my.bpt ...` |
| `petey infer-schema ...` | `petey infer-blueprint ...` |
| `.yaml` blueprint files | `.bpt` (still parsed as YAML) |

Old names still work in v0.5.0 with a `DeprecationWarning` and will be removed in v0.6.0. To migrate existing `.yaml` blueprint files, just rename them — the file format is unchanged:

```bash
find . -name "*.yaml" -exec sh -c 'mv "$1" "${1%.yaml}.bpt"' _ {} \;
```

## Optional Dependencies

```bash
pip install petey                    # Core (pymupdf, pdfplumber, openai, anthropic, litellm)
pip install petey[unstructured]      # + Unstructured API client
pip install petey[all]               # Everything
```

Direct backends with extra SDK requirements:

| Backend | Install |
|---------|---------|
| `gemini`, `vertex_ai` | `pip install google-genai` |
| `anthropic_bedrock`, `anthropic_vertex` | already covered by the core `anthropic` dep |
| `ollama` | none — uses Ollama's OpenAI-compatible endpoint |
