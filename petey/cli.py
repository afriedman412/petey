"""
CLI entrypoint for Petey.

Usage:
    petey extract --schema schema.yaml ./pdfs/*.pdf
    petey extract --schema schema.yaml ./pdfs/ -o results.csv
    petey extract --schema schema.yaml ./pdfs/ --format jsonl -o results.jsonl
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from petey.schema import load_schema
from petey.extract import extract_batch, extract_pages_async


def _collect_pdfs(paths: list[str]) -> list[str]:
    """Expand directories and globs into a flat list of PDF paths."""
    result = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            result.extend(str(f) for f in sorted(path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf" and path.exists():
            result.append(str(path))
    return result


def _flatten(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Flatten nested array fields (same logic as the web UI)."""
    flat_records = []
    all_keys = []
    key_set = set()

    for rec in records:
        flat = {}
        nested_items = None
        for k, v in rec.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                nested_items = v
            else:
                flat[k] = v
                if k not in key_set:
                    key_set.add(k)
                    all_keys.append(k)
        if nested_items:
            for item in nested_items:
                row = {**flat}
                for k, v in item.items():
                    row[k] = v
                    if k not in key_set:
                        key_set.add(k)
                        all_keys.append(k)
                flat_records.append(row)
        else:
            flat_records.append(flat)

    return flat_records, all_keys


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="petey",
        description="Petey — The Easy PDF Extractor",
    )
    sub = parser.add_subparsers(dest="command")

    ext = sub.add_parser("extract", help="Extract data from PDFs")
    ext.add_argument("paths", nargs="+", help="PDF files or directories")
    ext.add_argument("--schema", "-s", required=True, help="YAML schema file")
    ext.add_argument(
        "--model", "-m", default=None,
        help="Model ID (default: gpt-4.1-mini)",
    )
    ext.add_argument(
        "--concurrency", "-c", type=int, default=10,
        help="Concurrent requests (default: 10)",
    )
    ext.add_argument(
        "--output", "-o", default=None,
        help="Output file (.csv, .json, or .jsonl)",
    )
    ext.add_argument(
        "--format", "-f", choices=["csv", "json", "jsonl"], default=None,
        help="Output format (inferred from -o extension if not set)",
    )
    ext.add_argument(
        "--instructions", "-i", default="",
        help="Additional extraction instructions",
    )
    ext.add_argument(
        "--pages-per-chunk", "-p", type=int, default=None,
        help="Pages per chunk (default: 1 for table schemas, 0 to disable)",
    )
    ext.add_argument(
        "--parser", default="pymupdf",
        choices=["pymupdf", "marker", "aryn"],
        help="PDF parser backend (default: pymupdf)",
    )
    ext.add_argument(
        "--ocr-fallback", action="store_true",
        help="Fall back to pytesseract OCR if pymupdf finds no text",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        run_extract(args)


def run_extract(args):
    response_model, spec = load_schema(args.schema)
    pdfs = _collect_pdfs(args.paths)

    if not pdfs:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    model = args.model or os.environ.get("PETEY_MODEL", "gpt-4.1-mini")
    is_array = spec.get("record_type") == "array"

    # Determine output format
    fmt = args.format
    if not fmt and args.output:
        ext = Path(args.output).suffix.lower()
        fmt = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}.get(
            ext, "jsonl"
        )
    if not fmt:
        fmt = "jsonl"

    # Streaming output for jsonl when writing to stdout
    out_file = None
    if fmt == "jsonl" and args.output:
        out_file = open(args.output, "w")

    completed = 0
    total = len(pdfs)

    def on_result(path, data):
        nonlocal completed
        completed += 1
        name = os.path.basename(path)
        if data.get("_error"):
            print(
                f"  [{completed}/{total}] ERROR {name}: {data['_error']}",
                file=sys.stderr,
            )
        else:
            print(f"  [{completed}/{total}] {name}", file=sys.stderr)
        # Stream JSONL immediately
        if fmt == "jsonl":
            line = json.dumps(data)
            if out_file:
                out_file.write(line + "\n")
                out_file.flush()
            else:
                print(line)

    instructions = args.instructions or spec.get("instructions", "")

    # Auto-chunk by page for array/table schemas unless explicitly disabled
    pages_per_chunk = args.pages_per_chunk
    if pages_per_chunk is None and is_array:
        pages_per_chunk = 1

    if pages_per_chunk:
        # Page-chunked mode: split each PDF into chunks, extract concurrently
        all_results = []
        for pdf in pdfs:
            from petey.extract import extract_text_pages
            n_pages = len(extract_text_pages(pdf))
            n_chunks = -(-n_pages // pages_per_chunk)  # ceil division
            pdf_name = os.path.basename(pdf)
            print(
                f"Petey: splitting {pdf_name} into {n_chunks} chunk(s) "
                f"({pages_per_chunk} page(s) each, {n_pages} pages total) "
                f"with {model} (concurrency={args.concurrency})",
                file=sys.stderr,
            )
            chunk_completed = 0

            def on_chunk(label, data, _name=pdf_name):
                nonlocal chunk_completed
                chunk_completed += 1
                if data.get("_error"):
                    print(
                        f"  [{chunk_completed}/{n_chunks}] ERROR "
                        f"{_name} {label}: {data['_error']}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  [{chunk_completed}/{n_chunks}] {_name} {label}",
                        file=sys.stderr,
                    )
                data["_source_file"] = _name
                if fmt == "jsonl":
                    line = json.dumps(data)
                    if out_file:
                        out_file.write(line + "\n")
                        out_file.flush()
                    else:
                        print(line)

            results = asyncio.run(
                extract_pages_async(
                    pdf, response_model,
                    model=model,
                    instructions=instructions,
                    pages_per_chunk=pages_per_chunk,
                    concurrency=args.concurrency,
                    on_result=on_chunk,
                    parser=args.parser,
                )
            )
            all_results.extend(results)
        results = all_results
    else:
        # Standard multi-file mode
        print(
            f"Petey: extracting {total} file{'s' if total > 1 else ''} "
            f"with {model} (concurrency={args.concurrency})",
            file=sys.stderr,
        )

        results = asyncio.run(
            extract_batch(
                pdfs, response_model,
                model=model,
                instructions=instructions,
                concurrency=args.concurrency,
                on_result=on_result,
                parser=args.parser,
                ocr_fallback=args.ocr_fallback,
            )
        )

    if out_file:
        out_file.close()

    # Unwrap array results
    all_records = []
    for data in results:
        if is_array and "items" in data:
            for item in data["items"]:
                item["_source_file"] = data.get("_source_file", "")
                if "_page" in data:
                    item["_page"] = data["_page"]
                all_records.append(item)
        elif not data.get("_error"):
            all_records.append(data)

    if fmt == "csv":
        flat, keys = _flatten(all_records)
        if args.output:
            with open(args.output, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(flat)
            print(
                f"Wrote {len(flat)} rows to {args.output}", file=sys.stderr
            )
        else:
            w = csv.DictWriter(
                sys.stdout, fieldnames=keys, extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(flat)
    elif fmt == "json":
        output = json.dumps(all_records, indent=2)
        if args.output:
            Path(args.output).write_text(output)
            print(
                f"Wrote {len(all_records)} records to {args.output}",
                file=sys.stderr,
            )
        else:
            print(output)
    elif fmt == "jsonl" and not args.output:
        pass  # Already streamed in on_result

    print(
        f"Done. {len(all_records)} records from "
        f"{total} file{'s' if total > 1 else ''}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
