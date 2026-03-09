# Petey

The Easy PDF Extractor.

```bash
pip install petey
```

## Setup

Add your API key to a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Or for Anthropic:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
petey extract --schema schema.yaml ./pdfs/ -o results.csv
```

Options: `--model/-m` (default: `gpt-4.1-mini`), `--concurrency/-c` (default: 10), `--format/-f` (csv/json/jsonl), `--output/-o`, `--instructions/-i`.

## Schema

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

Field types: `string`, `number`, `date`, `enum` (with or without `values`), `array` (with nested `fields`).

All fields are nullable — the LLM returns `null` for anything it can't find.

Set `record_type: array` at the top level for table extraction (multiple records per document).

Add `instructions` at the top level to append guidance to the system prompt.
