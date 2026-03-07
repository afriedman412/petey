"""
PAR Decision Extractor using LLM (OpenAI) with few-shot prompting.
Extracts structured fields from DHCR PAR decision PDFs.

Pipeline:
  1. Extract text from PDF (PyMuPDF)
  2. LLM extracts all 14 fields (few-shot prompt)
  3. Validate fields with known formats (dockets, dates, enums)
  4. Re-query LLM for any fields that failed validation
"""
import re
import fitz
import json
import sys
import asyncio
from datetime import datetime
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# Load API key from sibling project's .env
load_dotenv(Path(__file__).parent.parent / "municipal-budget-rag" / ".env")

client = OpenAI()
async_client = AsyncOpenAI()

SYSTEM_PROMPT = """You extract structured data from DHCR PAR (Petition for Administrative Review) decisions. Return ONLY a JSON object with the keys described below.

These documents follow a standard structure:

HEADER (top of first page):
- Caption box with petitioner name (or anonymous placeholder like "PETITIONER X")
- "ADMINISTRATIVE REVIEW DOCKET NO." — this is the PAR docket (adm_review_docket)
- "RENT ADMINISTRATOR'S DOCKET NO." — this is the RA docket (ra_docket), a DIFFERENT number
- Owner/Tenant labels with party names

FIRST PARAGRAPH (after the title):
- PAR filing date ("On [DATE], the petitioner filed...")
- RA order date ("against an order issued on [DATE]")
- Address and apartment
- Description of what the RA decided (ra_determination)
- Sometimes the RA complaint filing date ("This proceeding was commenced on [DATE]")

FINAL SECTION:
- "THEREFORE... ORDERED, that this petition be... [denied/granted/etc.]" — the PAR determination
- "ISSUED:" stamp with the decision date (may appear on any page, often OCR-garbled)

Extract these fields:

{
  "petitioner": "Person or company name from the caption. Use null if only a generic label like 'PETITIONER X' appears.",
  "petitioner_type": "Owner or Tenant",
  "other_party": "The opposing party's actual name from the caption, or null if none given. Labels like 'Owner:' or 'Tenant:' without a name = null.",
  "adm_review_docket": "The PAR docket number from the header",
  "ra_docket": "The RA docket number(s) from the header — NOT the PAR docket. If multiple dockets appear (in parentheses, with 'RECONSID.', 'incorporating', etc.), include ALL as comma-separated. If a sequential range is given (e.g., 'YE410147S through YE410150S'), expand to all numbers in the range.",
  "address": "Full street address including number, street name, and borough/city",
  "apartment": "Apartment number, or 'Various' for building-wide cases, or null",
  "determination": "PAR outcome: Denied, Granted, Granted in Part, Dismissed, Revoked, Modified, Rescinded, Remanded, or Terminated",
  "ra_determination": "What the RA originally decided: Granted, Denied, Granted in Part, Terminated, or null",
  "par_filed_date": "YYYY-MM-DD or null",
  "ra_order_issued": "YYYY-MM-DD or null",
  "ra_case_filed": "YYYY-MM-DD or null",
  "issue_date": "From the 'ISSUED:' stamp, YYYY-MM-DD or null. If the stamp is garbled or the year looks implausible, return null. Cross-check against other dates in the document."
}

Important notes:
- These are OCR'd documents. Docket numbers should be copied exactly as printed — do not correct or modify them.
- OCR often separates digits in ISSUED stamps: "NOV 1 2 2013" = November 12, "APR 3 0 2014" = April 30.
- Use null when information is absent or unreadable — never use "Unknown", "N/A", or empty string.
- For ra_determination: "denied the complaint" / "no overcharge found" = Denied. "granted a rent reduction" = Granted.
- Return ONLY the JSON object, no explanation."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """Extract fields from this PAR decision:

STATE OF NEW YORK DIVISION OF HOUSING AND COMMUNITY RENEWAL OFFICE OF RENT ADMINISTRATION GERTZ PLAZA 92-31 UNION HALL STREET JAMAICA, NEW YORK 11433

IN THE MATTER OF THE ADMINISTRATIVE APPEAL OF
DOMINICK VALENTINO, PETITIONER

ADMINISTRATIVE REVIEW DOCKET NO.: ZG410017RT
RENT ADMINISTRATOR'S DOCKET NO.: YD410048R
OWNER: BCRE WEST 72 LLC and STELLAR 85, LLC
TENANT OF RECORD: DAVID VALENTINO

ORDER AND OPINION DENYING PETITION FOR ADMINISTRATIVE REVIEW

On April 9, 2010, the above-named petitioner-tenant filed a rent overcharge complaint concerning the housing accommodations known as Room 1609 in the Hotel Olcott located at 27 West 72nd Street in Manhattan.

On June 10, 2011, the Rent Administrator issued an Order Denying Application or Terminating Proceeding finding that, "...the tenant paid the same rental amount of $1100.00 from the base date through the present. Therefore, no overcharge is found."

On July 7, 2011, said petitioner-tenant timely filed a Petition for Administrative Review (PAR) against the above-referenced Rent Administrator's order.

[...body text about four-year rule, Thornton v. Baron, Grimm v. DHCR...]

THEREFORE, pursuant to the applicable statutes and regulations, it is ORDERED, that this PAR be, and the same hereby is, denied and that the Rent Administrator's order be, and the same hereby is, affirmed.

ISSUED: NOV 1 2 2013"""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "petitioner": "Dominick Valentino",
            "petitioner_type": "Tenant",
            "other_party": "BCRE West 72 LLC and Stellar 85, LLC",
            "adm_review_docket": "ZG410017RT",
            "ra_docket": "YD410048R",
            "address": "27 West 72nd Street, Manhattan",
            "apartment": "1609",
            "determination": "Denied",
            "ra_determination": "Denied",
            "par_filed_date": "2011-07-07",
            "ra_order_issued": "2011-06-10",
            "ra_case_filed": "2010-04-09",
            "issue_date": "2013-11-12"
        }, indent=2)
    },
    {
        "role": "user",
        "content": """Extract fields from this PAR decision:

STATE OF NEW YORK DIVISION OF HOUSING AND COMMUNITY RENEWAL OFFICE OF RENT ADMINISTRATION GERTZ PLAZA 92-31 UNION HALL STREET JAMAICA, NEW YORK 11433

IN THE MATTER OF THE ADMINISTRATIVE APPEAL OF
ERIC WARNER, PETITIONER

SJR NO. 14026
ADMINISTRATIVE REVIEW DOCKET NO. ZE420003RK
RENT ADMINISTRATOR'S DOCKET NO. WD420001LD
OWNER: MAYFLOWER DEVELOPMENT CORP.

ORDER AND OPINION DENYING FURTHER RECONSIDERATION ON PETITION FOR ADMINISTRATIVE REVIEW

On June 4, 2008, the above-named tenant filed a petition for administrative review (PAR) of an order issued on May 29, 2008 by a Rent Administrator concerning the housing accommodations known as 425 Riverside Drive, Apartment 161, New York, New York, wherein the Rent Administrator determined that the subject apartment was to be deregulated effective June 1, 2008, pursuant to Section 2200.2(f)(20) of the New York City Rent and Eviction Regulations.

[...body text about luxury decontrol, J-51 benefits, Roberts v. Tishman Speyer...]

THEREFORE, pursuant to the provisions of the applicable statutes and regulations, it is ORDERED, that this petition for administrative review be granted and that the Rent Administrator's order be modified in accordance with this order and opinion and affirmed in all other respects.

ISSUED: DEC 0 7 2011"""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "petitioner": "Eric Warner",
            "petitioner_type": "Tenant",
            "other_party": "Mayflower Development Corp.",
            "adm_review_docket": "ZE420003RK",
            "ra_docket": "WD420001LD",
            "address": "425 Riverside Drive, New York, New York",
            "apartment": "161",
            "determination": "Granted",
            "ra_determination": "Granted",
            "par_filed_date": "2008-06-04",
            "ra_order_issued": "2008-05-29",
            "ra_case_filed": None,
            "issue_date": "2011-12-07"
        }, indent=2)
    },
    {
        "role": "user",
        "content": """Extract fields from this PAR decision:

STATE OF NEW YORK DIVISION OF HOUSING AND COMMUNITY RENEWAL OFFICE OF RENT ADMINISTRATION GERTZ PLAZA 92-31 UNION HALL STREET JAMAICA, NEW YORK 11433

IN THE MATTER OF THE ADMINISTRATIVE APPEAL OF

PETITIONER X

ADMINISTRATIVE REVIEW DOCKET NO.: YH610037RO
RENT ADMINISTRATOR'S DOCKET NO.: YD610025S

ORDER AND OPINION DENYING PETITION FOR ADMINISTRATIVE REVIEW

On August 21, 2010, the above-named petitioner-owner filed a Petition for Administrative Review (PAR) of an order the Rent Administrator issued on July 20, 2010 concerning the housing accommodations known as 1639 Nelson Avenue, Apt 3F, Bronx, NY, wherein the Administrator, based upon the entire record, directed the restoration of services and granted a rent reduction.

[...body text about owner's claims regarding rent regulation status...]

THEREFORE, in accordance with the relevant Rent Regulatory Laws and Regulations, it is ORDERED, that this petition be, and the same hereby is, denied, and that the Rent Administrator's order be, and the same hereby is, affirmed.

ISSUED: JAN 0 5 2011"""
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "petitioner": None,
            "petitioner_type": "Owner",
            "other_party": None,
            "adm_review_docket": "YH610037RO",
            "ra_docket": "YD610025S",
            "address": "1639 Nelson Avenue, Bronx, NY",
            "apartment": "3F",
            "determination": "Denied",
            "ra_determination": "Granted",
            "par_filed_date": "2010-08-21",
            "ra_order_issued": "2010-07-20",
            "ra_case_filed": None,
            "issue_date": "2011-01-05"
        }, indent=2)
    }
]


def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n\n".join(pages)


def _build_extract_messages(text: str) -> list[dict]:
    """Build the message list for field extraction."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Extract fields from this PAR decision:\n\n{text}"},
    ]


def _build_requery_messages(text: str, errors: list[dict]) -> list[dict]:
    """Build the message list for re-query correction."""
    error_descriptions = [
        f"- {e['field']}: you returned {json.dumps(e['value'])}. Problem: {e['error']}"
        for e in errors
    ]
    correction_prompt = f"""Some extracted fields failed validation. Here are the problems:

{chr(10).join(error_descriptions)}

Hints:
- Docket format: 2 letters + 6 digits + 1-3 letter suffix. OCR can cause digit/letter confusion (0↔O, 1↔I, 5↔S, 8↔B). Only fix the original value — do not substitute a different docket from the document.
- The RA docket is labeled "RENT ADMINISTRATOR'S DOCKET NO." in the header. It is NOT the same as the PAR/Administrative Review docket.
- If a date's year seems implausible given other dates in the document, return null rather than guessing.

Re-read the document and return corrected values as JSON (only the fields that need correction):

{text}"""
    return [
        {"role": "system", "content": "You are correcting specific extracted fields from a DHCR PAR decision. Return ONLY a JSON object with the corrected field values."},
        {"role": "user", "content": correction_prompt},
    ]


LLM_PARAMS = dict(temperature=0, response_format={"type": "json_object"})


def extract_fields(text: str, model: str = "gpt-4o-mini") -> dict:
    """Send document text to OpenAI and extract structured fields."""
    response = client.chat.completions.create(
        model=model, messages=_build_extract_messages(text), **LLM_PARAMS,
    )
    return json.loads(response.choices[0].message.content)


# --- Validation ---

# Docket: 2 letters + 6 digits + 1-3 letter suffix
DOCKET_RE = re.compile(r'^[A-Z]{2}\d{6}[A-Z]{1,3}$')

# PAR docket must end with RO, RT, RK, or RP
PAR_SUFFIX_RE = re.compile(r'^[A-Z]{2}\d{6}(RO|RT|RK|RP)$')

# Known RA suffixes
VALID_RA_SUFFIXES = {
    'S', 'R', 'OM', 'HW', 'B', 'TA', 'LD', 'FR', 'RV',
    'OR', 'OE', 'TE', 'TH', 'OH', 'SS', 'OS', 'TS',
}

# Extract suffix: 2 letters + 6 digits + suffix
DOCKET_SUFFIX_RE = re.compile(r'^[A-Z]{2}\d{6}([A-Z]+)$')


def derive_case_type(ra_docket: str | None) -> str | None:
    """Derive ra_case_type from the ra_docket suffix. Uses the first docket if multiple."""
    if not ra_docket:
        return None
    # Take first docket if comma-separated
    first = ra_docket.split(',')[0].strip()
    m = DOCKET_SUFFIX_RE.match(first)
    if m:
        return m.group(1)
    return None

VALID_DETERMINATIONS = {
    'Denied', 'Granted', 'Granted in Part', 'Dismissed',
    'Revoked', 'Modified', 'Rescinded', 'Remanded', 'Terminated',
}

VALID_RA_DETERMINATIONS = {
    'Granted', 'Denied', 'Granted in Part', 'Terminated', None,
}


def validate_docket(docket: str | None, is_par: bool = False) -> str | None:
    """Check if a docket matches the expected format. Returns error message or None."""
    if docket is None:
        return "docket is null"
    if is_par:
        if not PAR_SUFFIX_RE.match(docket):
            return (
                f"'{docket}' doesn't match PAR docket format "
                f"(expected: 2 letters + 6 digits + RO/RT/RK/RP)"
            )
    else:
        if not DOCKET_RE.match(docket):
            return (
                f"'{docket}' doesn't match docket format "
                f"(expected: 2 letters + 6 digits + 1-3 letter suffix)"
            )
        if PAR_SUFFIX_RE.match(docket):
            return (
                f"'{docket}' ends in RO/RT/RK/RP — that's a PAR docket, "
                f"not an RA docket. Find the Rent Administrator's docket instead."
            )
    return None


def validate_date(date_str: str | None, field_name: str) -> str | None:
    """Check if a date string is valid YYYY-MM-DD. Returns error or None."""
    if date_str is None:
        return None
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return None
    except ValueError:
        return f"'{date_str}' for {field_name} is not valid YYYY-MM-DD"


def docket_year(docket: str | None) -> int | None:
    """Estimate filing year from a PAR docket's year+month codes."""
    if not docket or len(docket) < 2:
        return None
    year_code = docket[0].upper()
    month_code = docket[1].upper()
    if not year_code.isalpha() or not month_code.isalpha():
        return None
    base_year = ord(year_code) - ord('A')
    if month_code >= 'M':  # cycle 2 (2012-2037)
        return 2012 + base_year
    else:  # cycle 1 (1986-2011)
        return 1986 + base_year


def validate_result(result: dict) -> list[dict]:
    """Validate extracted fields. Returns list of {field, value, error} dicts."""
    errors = []

    err = validate_docket(result.get('adm_review_docket'), is_par=True)
    if err:
        errors.append({'field': 'adm_review_docket', 'value': result.get('adm_review_docket'), 'error': err})

    ra_docket_val = result.get('ra_docket')
    if ra_docket_val:
        # May be comma-separated multiple dockets
        for d in ra_docket_val.split(','):
            d = d.strip()
            if d:
                err = validate_docket(d, is_par=False)
                if err:
                    errors.append({'field': 'ra_docket', 'value': ra_docket_val, 'error': err})
                    break
    else:
        errors.append({'field': 'ra_docket', 'value': ra_docket_val, 'error': 'docket is null'})

    det = result.get('determination')
    if det and det not in VALID_DETERMINATIONS:
        errors.append({'field': 'determination', 'value': det, 'error': f"'{det}' is not a valid determination"})

    ra_det = result.get('ra_determination')
    if ra_det not in VALID_RA_DETERMINATIONS:
        errors.append({'field': 'ra_determination', 'value': ra_det, 'error': f"'{ra_det}' is not a valid RA determination"})

    for date_field in ['par_filed_date', 'ra_order_issued', 'ra_case_filed', 'issue_date']:
        err = validate_date(result.get(date_field), date_field)
        if err:
            errors.append({'field': date_field, 'value': result.get(date_field), 'error': err})

    # Issue date year plausibility check
    issue_date = result.get('issue_date')
    if issue_date:
        try:
            year = int(issue_date[:4])
            if year < 2005 or year > 2025:
                errors.append({'field': 'issue_date', 'value': issue_date,
                               'error': f"year {year} is outside plausible range (2005-2025)"})
        except (ValueError, TypeError):
            pass

    # Cross-check: issue_date should be after par_filed_date
    par_filed = result.get('par_filed_date')
    if issue_date and par_filed:
        try:
            issue_dt = datetime.strptime(issue_date, '%Y-%m-%d')
            par_dt = datetime.strptime(par_filed, '%Y-%m-%d')
            if issue_dt < par_dt:
                errors.append({'field': 'issue_date', 'value': issue_date,
                               'error': f"issue_date {issue_date} is before par_filed_date {par_filed}"})
        except ValueError:
            pass

    # Cross-check: issue_date year should be within reasonable range of PAR docket year
    par_docket = result.get('adm_review_docket')
    par_yr = docket_year(par_docket)
    if issue_date and par_yr:
        try:
            issue_yr = int(issue_date[:4])
            if issue_yr < par_yr or issue_yr > par_yr + 7:
                errors.append({'field': 'issue_date', 'value': issue_date,
                               'error': f"issue_date year {issue_yr} is implausible for "
                               f"PAR docket {par_docket} (filed ~{par_yr}, "
                               f"expected {par_yr}-{par_yr+7}). "
                               f"OCR may have garbled the year — if unreadable, return null."})
        except (ValueError, TypeError):
            pass

    return errors


# --- Re-query for failed fields ---

def requery_fields(text: str, result: dict, errors: list[dict],
                   model: str = "gpt-4o-mini") -> dict:
    """Ask the LLM to correct specific fields that failed validation."""
    response = client.chat.completions.create(
        model=model, messages=_build_requery_messages(text, errors), **LLM_PARAMS,
    )
    return json.loads(response.choices[0].message.content)


# --- Main pipeline ---

def _coerce_list_values(result: dict) -> None:
    """Ensure all values are strings or None (LLM sometimes returns lists)."""
    for key in list(result.keys()):
        if isinstance(result[key], list):
            result[key] = ", ".join(str(v) for v in result[key])


def _apply_corrections(result: dict, corrections: dict, error_fields: list[str],
                       label: str = "") -> None:
    """Apply re-query corrections to the result dict."""
    for field, value in corrections.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        if field in error_fields:
            old = result[field]
            result[field] = value
            print(f"  {label}corrected {field}: {old!r} -> {value!r}")


def _finalize_result(result: dict, pdf_path: str, text_length: int) -> dict:
    """Add derived fields and metadata to the result."""
    result['ra_case_type'] = derive_case_type(result.get('ra_docket'))
    result["_source_file"] = Path(pdf_path).name
    result["_text_length"] = text_length
    return result


def _handle_validation(result: dict, errors: list[dict]) -> None:
    """Check remaining validation errors and attach warnings."""
    remaining = validate_result(result)
    if remaining:
        result['_validation_warnings'] = [
            f"{e['field']}: {e['error']}" for e in remaining
        ]


def process_file(pdf_path: str, model: str = "gpt-4o-mini") -> dict:
    """Extract text from PDF, run LLM extraction, validate, and re-query if needed."""
    text = extract_text(pdf_path)
    result = extract_fields(text, model=model)
    _coerce_list_values(result)

    errors = validate_result(result)
    if errors:
        error_fields = [e['field'] for e in errors]
        print(f"  Validation errors: {error_fields}")
        corrections = requery_fields(text, result, errors, model=model)
        _apply_corrections(result, corrections, error_fields)
        _handle_validation(result, errors)

    return _finalize_result(result, pdf_path, len(text))


async def async_extract_fields(text: str, model: str = "gpt-4o-mini") -> dict:
    """Async version of extract_fields."""
    response = await async_client.chat.completions.create(
        model=model, messages=_build_extract_messages(text), **LLM_PARAMS,
    )
    return json.loads(response.choices[0].message.content)


async def async_requery_fields(text: str, result: dict, errors: list[dict],
                               model: str = "gpt-4o-mini") -> dict:
    """Async version of requery_fields."""
    response = await async_client.chat.completions.create(
        model=model, messages=_build_requery_messages(text, errors), **LLM_PARAMS,
    )
    return json.loads(response.choices[0].message.content)


async def async_process_file(pdf_path: str, model: str = "gpt-4o-mini") -> dict:
    """Async version of process_file."""
    text = extract_text(pdf_path)
    result = await async_extract_fields(text, model=model)
    _coerce_list_values(result)

    errors = validate_result(result)
    if errors:
        error_fields = [e['field'] for e in errors]
        label = f"{Path(pdf_path).name}: "
        print(f"  {label}validation errors: {error_fields}")
        corrections = await async_requery_fields(text, result, errors, model=model)
        _apply_corrections(result, corrections, error_fields, label=label)
        _handle_validation(result, errors)

    return _finalize_result(result, pdf_path, len(text))


def main():
    par_dir = Path(__file__).parent / "PAR_files"

    if len(sys.argv) > 1:
        files = [par_dir / f for f in sys.argv[1:]]
    else:
        files = sorted(par_dir.glob("*.pdf"))[:1]

    for pdf_path in files:
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            continue

        print(f"\nProcessing: {pdf_path.name}")
        print("-" * 60)

        result = process_file(str(pdf_path))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
