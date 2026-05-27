"""
CLI entrypoint for Petey.

Usage:
    petey extract --blueprint blueprint.bpt ./pdfs/*.pdf
    petey extract --blueprint blueprint.bpt ./pdfs/ -o results.csv
    petey extract --blueprint blueprint.bpt ./pdfs/ --format jsonl -o results.jsonl
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import warnings
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

import yaml

from petey.schema import load_blueprint, normalize_dates
from petey.extract import (
    extract_batch, extract_pages_async, infer_blueprint,
    PARSERS, LLM_BACKENDS,
    API_PARSERS, PLUGIN_PARSERS,
    PLUGIN_LLM_BACKENDS,
)
from petey.registries.models import (
    MODELS, load_models_file, register_models, model_source,
    user_models_path, DEFAULT_USER_MODELS_PATH, default_model,
)


_USER_MODELS_TEMPLATE = """\
# Petey user model registry. Entries here augment the built-in
# MODELS registry; on key collision, this file wins. Each entry:
#
#   <name>:
#     provider: <backend name from `petey list llm`>
#     model:    <wire-format model name on the provider, if it
#                differs from <name>>
#     config:   <optional kwargs forwarded to the backend builder>
#     kwargs:   <optional kwargs forwarded to chat.completions.create>
#
# Run `petey models list` to see what's currently registered.

# Example — Azure OpenAI deployment:
#
# my-azure-gpt-4o:
#   provider: azure_openai
#   model: gpt-4o          # the deployment name on Azure
#   config:
#     api_version: "2024-06-01"
#     azure_endpoint: https://your-tenant.openai.azure.com
#     api_key_env: MY_AZURE_KEY
#     organization: "12345"

# Example — custom Ollama host:
#
# remote-qwen:
#   provider: ollama
#   model: qwen2.5:7b
#   config:
#     base_url: http://gpu-box.local:11434/v1
"""


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
    # find_dotenv(usecwd=True) walks up from the user's working
    # directory; the default walks up from this module's location,
    # which makes `petey` pick up a .env from wherever petey itself
    # is installed.
    load_dotenv(find_dotenv(usecwd=True))

    parser = argparse.ArgumentParser(
        prog="petey",
        description="Petey — The Easy PDF Extractor",
    )
    sub = parser.add_subparsers(dest="command")

    ext = sub.add_parser("extract", help="Extract data from PDFs")
    ext.add_argument(
        "paths", nargs="*",
        help="PDF files or directories (overrides input in blueprint)",
    )
    _bp_group = ext.add_mutually_exclusive_group(required=True)
    _bp_group.add_argument(
        "--blueprint", "-b",
        help="Blueprint file (.bpt or .yaml)",
    )
    _bp_group.add_argument(
        "--schema", "-s",
        help=argparse.SUPPRESS,  # deprecated; use --blueprint
    )
    ext.add_argument(
        "--model", "-m", default=None,
        help="Model ID (default: gpt-4.1-mini)",
    )
    ext.add_argument(
        "--llm-backend", default=None,
        help=(
            "Override the LLM backend (e.g. azure_openai). Useful "
            "when the model name's default backend isn't right — "
            "e.g. running gpt-4o through Azure rather than openai. "
            "Backend reads its config from env vars."
        ),
    )
    ext.add_argument(
        "--models-config", default=None,
        help=(
            "Path to a YAML file of model registry entries to merge "
            "for this run only (in addition to the built-in registry "
            "and ~/.petey/models.yaml)."
        ),
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
        "--pages-per-chunk", "-p", type=int, default=None,
        help="Pages per chunk (default: 1 for table schemas, 0 to disable)",
    )
    ext.add_argument(
        "--parser", default=None,
        choices=list(PARSERS.keys()),
        help=(
            "PDF text extraction backend "
            "(default: from schema, or pymupdf)"
        ),
    )
    mode_group = ext.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode", choices=["query", "table"], default=None,
        help=(
            "Extraction mode: query (one record per file) "
            "or table (multiple records per page). "
            "Overrides mode in the schema."
        ),
    )
    mode_group.add_argument(
        "--query", action="store_const", const="query", dest="mode",
        help="Shorthand for --mode query",
    )
    mode_group.add_argument(
        "--table", action="store_const", const="table", dest="mode",
        help="Shorthand for --mode table",
    )
    ext.add_argument(
        "--header-pages", type=int, default=None,
        help="Header pages to prepend to each chunk",
    )
    ext.add_argument(
        "--page-range", default=None,
        help="Page range to extract (e.g. 2-5 or 1,3,5-7)",
    )

    # --- infer-blueprint subcommand (alias: infer-schema, deprecated) ---
    inf = sub.add_parser(
        "infer-blueprint",
        aliases=["infer-schema"],
        help="Suggest a blueprint from a sample PDF",
    )
    inf.add_argument("pdf", help="PDF file to analyze")
    inf.add_argument(
        "--model", "-m", default=None,
        help="Model ID (default: gpt-4.1-mini)",
    )
    inf.add_argument(
        "--llm-backend", default=None,
        help="Override the LLM backend (see `extract --llm-backend`).",
    )
    inf.add_argument(
        "--models-config", default=None,
        help="Path to a YAML file of model registry entries to merge "
             "for this run only.",
    )
    inf.add_argument(
        "--max-pages", type=int, default=2,
        help="Pages to sample (default: 2)",
    )
    inf.add_argument(
        "--parser", default="pymupdf",
        choices=list(PARSERS.keys()),
        help="PDF text extraction backend",
    )
    inf.add_argument(
        "--output", "-o", default=None,
        help="Save blueprint to file (.bpt recommended)",
    )

    # --- list subcommand ---
    lst = sub.add_parser(
        "list",
        help="List available backends",
    )
    lst.add_argument(
        "backend", nargs="?", default="all",
        choices=["all", "parsers", "llm"],
        help="Which backends to list (default: all)",
    )

    # --- models subcommand ---
    mdl = sub.add_parser(
        "models",
        help="Inspect and manage the model registry",
    )
    mdl_sub = mdl.add_subparsers(dest="models_action")
    mdl_sub.add_parser(
        "list", help="List registered models with their source",
    )
    mdl_sub.add_parser(
        "path",
        help="Print the resolved user-config models path",
    )
    init_p = mdl_sub.add_parser(
        "init",
        help="Write a commented template to the user-config path "
             "(only if the file does not already exist)",
    )
    init_p.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing file. Otherwise refuses.",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        run_extract(args)
    elif args.command in ("infer-blueprint", "infer-schema"):
        if args.command == "infer-schema":
            warnings.warn(
                "The 'infer-schema' subcommand is deprecated; "
                "use 'infer-blueprint'. "
                "Support will be removed in v0.6.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        run_infer_blueprint(args)
    elif args.command == "list":
        run_list(args)
    elif args.command == "models":
        run_models(args)


def _apply_models_config(path: str | None) -> None:
    """Merge a per-run models YAML into MODELS, if a path is given."""
    if not path:
        return
    p = Path(path)
    if not p.exists():
        print(f"--models-config: file not found: {p}", file=sys.stderr)
        sys.exit(1)
    register_models(load_models_file(p), source=str(p))


def run_extract(args):
    _apply_models_config(args.models_config)
    if args.schema:
        warnings.warn(
            "The --schema flag is deprecated; use --blueprint. "
            "Support will be removed in v0.6.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    blueprint_path = args.blueprint or args.schema
    response_model, spec = load_blueprint(blueprint_path)

    # Input: CLI paths override blueprint input
    paths = args.paths or []
    if not paths and spec.get("input"):
        paths = [spec["input"]]
    if not paths:
        print(
            "No input specified. Provide paths on the command line "
            "or set 'input:' in the blueprint.",
            file=sys.stderr,
        )
        sys.exit(1)
    pdfs = _collect_pdfs(paths)

    if not pdfs:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    model = (
        args.model
        or spec.get("model")
        or os.environ.get("PETEY_MODEL")
        or default_model()
        or "gpt-4.1-mini"
    )
    from petey.extract import _get_provider
    try:
        provider = _get_provider(model, args.llm_backend)
    except Exception:
        provider = "unknown"

    # CLI --mode / --table / --query overrides schema mode
    if args.mode is not None:
        spec["mode"] = args.mode
    # Backwards compat: record_type: array → mode: table
    if spec.get("record_type") == "array" and "mode" not in spec:
        spec["mode"] = "table"
    is_table = spec.get("mode") == "table"

    # Output: CLI -o overrides schema output
    output_path = args.output or spec.get("output")

    # Determine output format
    fmt = args.format
    if not fmt and output_path:
        ext = Path(output_path).suffix.lower()
        fmt = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}.get(
            ext, "csv" if is_table else "json",
        )
    if not fmt:
        fmt = "csv" if is_table else "json"

    # Streaming output for jsonl when writing to stdout
    out_file = None
    if fmt == "jsonl" and output_path:
        out_file = open(output_path, "w")

    completed = 0
    total = len(pdfs)

    def on_result(path, data):
        nonlocal completed
        completed += 1
        name = os.path.basename(path)
        if data.get("_error"):
            print(
                f"  [{completed}/{total}] ERROR {name} "
                f"[model={model}, provider={provider}]: {data['_error']}",
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

    instructions = spec.get("instructions", "")
    parser_options = spec.get("parser_options") or None

    # Parser: CLI overrides schema; schema overrides default
    if args.parser is not None:
        parser = args.parser
    else:
        parser = spec.get("parser", "pymupdf")

    # Auto-chunk by page for array/table schemas unless explicitly disabled
    pages_per_chunk = args.pages_per_chunk
    if pages_per_chunk is None and is_table:
        pages_per_chunk = 1

    # Build a summary of non-default options for status messages
    extras = []
    if parser != "pymupdf":
        extras.append(f"parser={parser}")
    opts = f"concurrency={args.concurrency}"
    if extras:
        opts = ", ".join(extras) + ", " + opts

    if pages_per_chunk:
        # Page-chunked mode: split each PDF into chunks, extract concurrently
        all_results = []
        for pdf in pdfs:
            import fitz
            n_pages = len(fitz.open(pdf))
            n_chunks = -(-n_pages // pages_per_chunk)  # ceil division
            pdf_name = os.path.basename(pdf)
            print(
                f"Petey: splitting {pdf_name} into {n_chunks} chunk(s) "
                f"({pages_per_chunk} page(s) each, {n_pages} pages total) "
                f"with {model} (provider={provider}, {opts})",
                file=sys.stderr,
            )
            chunk_completed = 0

            def on_chunk(label, data, _name=pdf_name):
                nonlocal chunk_completed
                chunk_completed += 1
                if data.get("_error"):
                    print(
                        f"  [{chunk_completed}/{n_chunks}] ERROR "
                        f"{_name} {label} "
                        f"[model={model}, provider={provider}]: "
                        f"{data['_error']}",
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
                    llm_backend=args.llm_backend,
                    instructions=instructions,
                    pages_per_chunk=pages_per_chunk,
                    concurrency=args.concurrency,
                    on_result=on_chunk,
                    parser=parser,
                    parser_options=parser_options,
                    header_pages=(
                        args.header_pages
                        if args.header_pages is not None
                        else spec.get("header_pages", 0)
                    ),
                    page_range=(
                        args.page_range
                        or spec.get("pages")
                        or None
                    ),
                )
            )
            all_results.extend(results)
        results = all_results
    else:
        # Standard multi-file mode
        print(
            f"Petey: extracting {total} file{'s' if total > 1 else ''} "
            f"with {model} (provider={provider}, {opts})",
            file=sys.stderr,
        )

        results = asyncio.run(
            extract_batch(
                pdfs, response_model,
                model=model,
                llm_backend=args.llm_backend,
                instructions=instructions,
                concurrency=args.concurrency,
                on_result=on_result,
                parser=parser,
                parser_options=parser_options,
            )
        )

    if out_file:
        out_file.close()

    # Unwrap array results
    all_records = []
    for data in results:
        if is_table and "items" in data:
            for item in data["items"]:
                item["_source_file"] = data.get("_source_file", "")
                if "_page" in data:
                    item["_page"] = data["_page"]
                all_records.append(item)
        elif not data.get("_error"):
            all_records.append(data)

    # Normalize date fields to YYYY-MM-DD
    for rec in all_records:
        normalize_dates(rec, spec)

    if fmt == "csv":
        flat, keys = _flatten(all_records)
        if output_path:
            with open(output_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(flat)
            print(
                f"Wrote {len(flat)} rows to {output_path}", file=sys.stderr
            )
        else:
            w = csv.DictWriter(
                sys.stdout, fieldnames=keys, extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(flat)
    elif fmt == "json":
        json_output = json.dumps(all_records, indent=2)
        if output_path:
            Path(output_path).write_text(json_output)
            print(
                f"Wrote {len(all_records)} records to {output_path}",
                file=sys.stderr,
            )
        else:
            print(json_output)
    elif fmt == "jsonl" and not output_path:
        pass  # Already streamed in on_result

    print(
        f"Done. {len(all_records)} records from "
        f"{total} file{'s' if total > 1 else ''}.",
        file=sys.stderr,
    )


def run_infer_blueprint(args):
    _apply_models_config(args.models_config)
    model = (
        args.model
        or os.environ.get("PETEY_MODEL")
        or default_model()
        or "gpt-4.1-mini"
    )
    from petey.extract import _get_provider
    try:
        provider = _get_provider(model, args.llm_backend)
    except Exception:
        provider = "unknown"
    print(
        f"Petey: analyzing {args.pdf} with {model} "
        f"(provider={provider}, sampling {args.max_pages} page(s))",
        file=sys.stderr,
    )

    spec = infer_blueprint(
        args.pdf,
        model=model,
        llm_backend=args.llm_backend,
        parser=args.parser,
        max_pages=args.max_pages,
    )

    output = yaml.dump(
        spec, default_flow_style=False, sort_keys=False,
    )

    if args.output:
        Path(args.output).write_text(output)
        print(
            f"Blueprint saved to {args.output}",
            file=sys.stderr,
        )
    else:
        print(output)

    n_fields = len(spec.get("fields", {}))
    mode = spec.get("mode", "query")
    print(
        f"Suggested {n_fields} fields "
        f"(mode: {mode})",
        file=sys.stderr,
    )


def _backend_type(name, api_dict, plugin_dict):
    """Classify a backend as built-in, API, or plugin."""
    if name in plugin_dict:
        return "plugin"
    if name in api_dict:
        return "api"
    return "built-in"


def run_models(args):
    action = args.models_action or "list"
    if action == "path":
        path = user_models_path()
        print(path)
        if not path.exists():
            print(
                "(file does not exist — run `petey models init` "
                "to create a template)",
                file=sys.stderr,
            )
        return
    if action == "init":
        path = user_models_path()
        if path.exists() and not args.force:
            print(
                f"Refusing to overwrite existing file: {path}\n"
                f"Use --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_USER_MODELS_TEMPLATE)
        print(f"Wrote template to {path}", file=sys.stderr)
        return
    # list
    print()
    print(f"{'name':40s}  {'provider':20s}  source")
    print(f"{'-' * 40}  {'-' * 20}  ------")
    for name in sorted(MODELS):
        entry = MODELS[name]
        src = model_source(name)
        print(f"{name:40s}  {entry['provider']:20s}  {src}")
    print()


def run_list(args):
    sections = {
        "parsers": (PARSERS, API_PARSERS, PLUGIN_PARSERS),
        "llm": (LLM_BACKENDS, {}, PLUGIN_LLM_BACKENDS),
    }
    show = (
        list(sections.keys()) if args.backend == "all"
        else [args.backend]
    )
    for section in show:
        registry, api_dict, plugin_dict = sections[section]
        print(f"\n{section}:")
        for name in registry:
            kind = _backend_type(name, api_dict, plugin_dict)
            print(f"  {name:20s} ({kind})")
    print()


if __name__ == "__main__":
    main()
