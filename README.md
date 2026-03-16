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

## Setup

Add your API key to a `.env` file in your working directory:

```
OPENAI_API_KEY=sk-...
```

Or for Anthropic:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Petey defaults to `gpt-4.1-mini`, which is fast and cheap and handles most documents well. To use a different model, pass `--model` on the CLI or set a default in `.env`:

```
PETEY_MODEL=claude-sonnet-4-5
```

Any OpenAI or Anthropic model ID works. Step up to a larger model if you're seeing errors on complex or dense documents.

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
    type: enum
    values: [Paid, Unpaid, Overdue]
```

### Field types

| Type | Notes |
|------|-------|
| `string` | Any text value |
| `number` | Integer or decimal |
| `date` | Returns ISO 8601 format |
| `enum` | Constrained set of values; optionally list `values:` to enforce them |

All fields are nullable — Petey returns `null` for anything it can't find rather than guessing.

### Schema options

| Option | Description |
|--------|-------------|
| `record_type: array` | Extract multiple records per document instead of one |
| `instructions` | Extra guidance appended to the prompt (e.g. "ignore the summary row") |
| `header_pages` | Number of leading pages to treat as a document header (see below) |
| `pages` | Page range to process, e.g. `"2-5"` or `"1,3,5-7"` (1-indexed) |
| `parser` | Text extraction backend: `pymupdf` (default), `tables`, or `pdfplumber` |

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
    type: enum
    values: [Paid, Unpaid, Overdue]
```

```bash
petey extract --schema invoice.yaml ./invoices/ -o results.csv
```

### One table per file

Some documents contain a table of records — a bank statement, a schedule of assets, a list of transactions. Add `record_type: array` and Petey will return multiple rows per file. Petey splits the document into pages, processes each concurrently, and assembles the results in document order.

Documents like this often have important context on the first page — a header, column labels, the filer's name. The `header_pages` option tells Petey to prepend those pages to every page it sends to the LLM, so that context is always visible no matter how deep into the document it's looking.

```yaml
name: Transactions
record_type: array
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
| `--parser` | `pymupdf` | PDF text extraction backend |
| `--ocr-fallback` | off | Fall back to OCR if no text layer is found |

## Notes

**Who this is for.** Petey is aimed at someone who has a large collection of PDFs that all share the same format and wants a reliable, low-effort way to get the data out of them.

**Cheap and good, but not fast.** Every page goes through an LLM API call. That makes Petey accurate and cheap — a large batch costs a few dollars, not hundreds — but it is not fast in the way a pure parser would be. Processing thousands of documents takes time. Petey runs extractions concurrently to keep throughput up, but if raw speed is the constraint, an LLM-based approach may not be the right tool.

**Designed to be modular.** Petey is a framework as much as a tool. The PDF-to-text step, the OCR fallback, and the LLM backend are all swappable. The parser option (`pymupdf`, `tables`, `pdfplumber`) lets you choose how text is extracted before it ever reaches the model. The model flag lets you swap in any OpenAI or Anthropic model. The schema drives everything else. The goal is to make it easy to tune each layer independently as your documents and requirements change.
