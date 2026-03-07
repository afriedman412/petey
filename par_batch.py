"""
Batch processor: extract fields from PAR PDFs and compare to regex parser output.
"""
import csv
import re
import json
import sys
import asyncio
from pathlib import Path
from datetime import datetime

from par_rag_extract import process_file, async_process_file


OUTPUT_DIR = Path(__file__).parent / "output"
GROUND_TRUTH = Path(__file__).parent / "parsed_pars.csv"

# Map our field names -> regex parser column names
FIELD_MAP = {
    "adm_review_docket": "Adm. Rev. Docket",
    "petitioner": "Petitioner",
    "petitioner_type": "Petitioner Type",
    "other_party": "Other Party",
    "ra_docket": "RA Docket",
    "ra_case_type": "RA Case Type",
    "ra_determination": "RA Determination",
    "address": "Address",
    "apartment": "Apt. No.",
    "determination": "Determination",
    "par_filed_date": "PAR filed",
    "ra_order_issued": "RA order issued",
    "ra_case_filed": "RA Case Filed",
    "issue_date": "Issue Date",
}


def load_ground_truth() -> dict:
    """Load regex parser CSV indexed by Adm. Rev. Docket."""
    gt = {}
    with open(GROUND_TRUTH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            docket = row["Adm. Rev. Docket"].strip()
            gt[docket] = row
    return gt


def normalize_date(date_str: str | None) -> str | None:
    """Try to normalize various date formats to YYYY-MM-DD."""
    if not date_str or date_str.strip() == "":
        return None
    date_str = date_str.strip()

    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%B %d,%Y",
        "%b %d %Y",
        "%b %d, %Y",
        "%B %d %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    parts = date_str.split()
    if len(parts) == 3:
        try:
            return datetime.strptime(
                " ".join(parts), "%b %d %Y"
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return date_str


def normalize_address(addr: str) -> str:
    """Normalize address for fuzzy comparison: lowercase, strip apt info,
    normalize abbreviations, collapse whitespace."""
    if not addr:
        return ""
    a = addr.lower().strip()
    # Remove leading apartment/unit info that regex parser sometimes includes
    a = re.sub(
        r'^(?:apt\.?\s*|apartment\s+|unit\s+)[^,]*,?\s*'
        r'(?:at\s+(?:the\s+)?premises?\s+)?', '', a
    )
    a = re.sub(r'\bat\s+the\s+premises\s+', '', a)
    a = re.sub(r'\bpremises\s+', '', a)
    # Normalize state/borough abbreviations
    a = a.replace("new york, new york", "new york, ny")
    a = a.replace("new york, ny", "new york, ny")
    a = a.replace(", new york", ", ny")
    a = re.sub(r'\bbrooklyn\b', 'brooklyn', a)
    a = re.sub(r'\bbronx\b', 'bronx', a)
    # Collapse whitespace
    a = re.sub(r'\s+', ' ', a).strip()
    # Remove trailing zip codes for comparison
    a = re.sub(r'\s+\d{5}(-\d{4})?$', '', a)
    return a


def extract_street_number(addr: str) -> str | None:
    """Pull leading street number from an address."""
    m = re.match(r'^(\d+[-\d]*)', addr.strip())
    return m.group(1) if m else None


def compare_field(extracted, ground_truth, field_name: str) -> dict:
    """Compare a single field. Returns match info."""
    is_date = field_name in (
        "par_filed_date", "ra_order_issued", "ra_case_filed", "issue_date"
    )
    is_address = field_name == "address"

    ext = str(extracted) if extracted is not None else ""
    gt = str(ground_truth) if ground_truth is not None else ""

    if is_date:
        ext_norm = normalize_date(ext) or ""
        gt_norm = normalize_date(gt) or ""
    elif is_address:
        ext_norm = normalize_address(ext)
        gt_norm = normalize_address(gt)
    else:
        ext_norm = ext.strip().lower()
        gt_norm = gt.strip().lower()

    is_docket = field_name in ("ra_docket", "adm_review_docket")

    if ext_norm == "" and gt_norm == "":
        match = True
    elif is_docket and ext_norm and gt_norm:
        # Multi-docket: match if docket sets overlap
        ext_set = {d.strip() for d in ext_norm.split(",")}
        gt_set = {d.strip() for d in gt_norm.split(",")}
        match = bool(ext_set & gt_set)
    elif is_address and ext_norm and gt_norm:
        # Fuzzy: check if street numbers match and one contains the other
        ext_num = extract_street_number(ext_norm)
        gt_num = extract_street_number(gt_norm)
        if ext_num and gt_num and ext_num == gt_num:
            match = True
        else:
            match = ext_norm == gt_norm
    else:
        match = ext_norm == gt_norm

    return {
        "extracted": ext,
        "ground_truth": gt,
        "match": match,
    }


def build_row(result: dict, gt_row: dict | None, comp_fields: list,
              pdf_name: str) -> dict:
    """Build a CSV row from extraction result and ground truth."""
    row = {
        "source_file": pdf_name,
        "gt_found": "yes" if gt_row else "no",
    }
    for field in comp_fields:
        ext_val = result.get(field)
        gt_val = gt_row.get(FIELD_MAP[field]) if gt_row else None
        comp = compare_field(ext_val, gt_val, field)
        row[f"{field}_ext"] = comp["extracted"]
        row[f"{field}_gt"] = comp["ground_truth"]
        row[f"{field}_match"] = comp["match"]
    row["validation_warnings"] = "; ".join(
        result.get("_validation_warnings", [])
    )
    return row


def update_field_stats(row: dict, comp_fields: list, field_stats: dict,
                       has_gt: bool):
    """Update per-field match statistics from a comparison row."""
    if not has_gt:
        return
    for field in comp_fields:
        ext_empty = row[f"{field}_ext"].strip() == ""
        gt_empty = row[f"{field}_gt"].strip() == ""
        if ext_empty and gt_empty:
            field_stats[field]["both_empty"] += 1
        elif row[f"{field}_match"] is True or row[f"{field}_match"] == "True":
            field_stats[field]["match"] += 1
        elif not ext_empty and gt_empty:
            field_stats[field]["ext_only"] += 1
        elif ext_empty and not gt_empty:
            field_stats[field]["gt_only"] += 1
        else:
            field_stats[field]["mismatch"] += 1


async def process_batch(pdf_dir: str, limit: int = None,
                        model: str = "gpt-4o-mini",
                        skip_existing: bool = False,
                        concurrency: int = 5):
    """Process PDFs concurrently, extract, compare to regex parser output."""
    par_dir = Path(pdf_dir)
    files = sorted(par_dir.glob("*.pdf"))

    print(f"Loading regex parser output from {GROUND_TRUTH}...")
    gt = load_ground_truth()
    print(f"  {len(gt)} records loaded")
    OUTPUT_DIR.mkdir(exist_ok=True)

    csv_path = OUTPUT_DIR / "extraction_comparison.csv"
    comp_fields = list(FIELD_MAP.keys())

    csv_cols = ["source_file", "gt_found"]
    for f in comp_fields:
        csv_cols.extend([f"{f}_ext", f"{f}_gt", f"{f}_match"])
    csv_cols.append("validation_warnings")

    # Load existing results if resuming
    already_done = set()
    if skip_existing and csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                already_done.add(row["source_file"])
        print(f"  {len(already_done)} already processed, skipping")

    files = [f for f in files if f.name not in already_done]
    if limit:
        files = files[:limit]
    print(f"Processing {len(files)} files (concurrency={concurrency})...\n")

    skipped = []
    error_files = []
    results = []
    field_stats = {f: {"match": 0, "mismatch": 0, "both_empty": 0,
                       "ext_only": 0, "gt_only": 0}
                   for f in comp_fields}

    mode = "a" if skip_existing and already_done else "w"
    csvfile = open(csv_path, mode, newline="")
    writer = csv.DictWriter(csvfile, fieldnames=csv_cols)
    if mode == "w":
        writer.writeheader()

    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def process_one(pdf_path):
        nonlocal completed
        docket = pdf_path.stem
        gt_row = gt.get(docket)

        async with sem:
            try:
                result = await async_process_file(str(pdf_path), model=model)
            except Exception as e:
                completed += 1
                print(f"[{completed}/{len(files)}] {pdf_path.name} ERROR: {e}")
                error_files.append({"file": pdf_path.name, "error": str(e)})
                return None

        completed += 1
        gt_label = "" if gt_row else " (no gt)"
        print(f"[{completed}/{len(files)}] {pdf_path.name}{gt_label}")

        if not gt_row:
            skipped.append(docket)

        row = build_row(result, gt_row, comp_fields, pdf_path.name)
        update_field_stats(row, comp_fields, field_stats, bool(gt_row))
        writer.writerow(row)
        csvfile.flush()
        results.append(result)
        return result

    tasks = [process_one(f) for f in files]
    await asyncio.gather(*tasks)

    csvfile.close()

    n_with_gt = sum(1 for r in results
                    if gt.get(r.get("adm_review_docket")))

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(results)} extracted, "
          f"{n_with_gt} compared to regex output, "
          f"{len(skipped)} not in regex output, "
          f"{len(error_files)} errors")
    print(f"CSV: {csv_path}")

    print(f"\nField comparison (of {n_with_gt} with regex output):")
    print(f"{'Field':<25} {'Agree':>6} {'Disagree':>8} "
          f"{'LLM only':>8} {'Regex only':>10} {'Empty':>6}")
    print("-" * 70)
    for field in comp_fields:
        s = field_stats[field]
        print(f"{field:<25} {s['match']:>6} {s['mismatch']:>8} "
              f"{s['ext_only']:>8} {s['gt_only']:>10} "
              f"{s['both_empty']:>6}")

    if error_files:
        print(f"\nErrors:")
        for err in error_files:
            print(f"  {err['file']}: {err['error']}")

    if skipped:
        print(f"\nNot in regex output ({len(skipped)}):")
        for s in skipped[:20]:
            print(f"  {s}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped)-20} more")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="PAR_files")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already in the output CSV")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent API calls")
    args = parser.parse_args()

    asyncio.run(process_batch(
        pdf_dir=args.dir,
        limit=args.limit,
        model=args.model,
        skip_existing=args.skip_existing,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
