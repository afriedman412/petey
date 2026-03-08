"""
Generic PDF field extractor.
Reads a YAML schema, builds a Pydantic model, uses Instructor to extract.
"""
import enum
import fitz
import yaml
import instructor
from pathlib import Path
from pydantic import BaseModel, Field, create_model
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

SYSTEM = "You extract structured data from documents. Use null for missing or unreadable values."

client = instructor.from_openai(OpenAI())
async_client = instructor.from_openai(AsyncOpenAI())

SCHEMAS_DIR = BASE_DIR / "schemas"


def _build_field(name: str, cfg: dict) -> tuple:
    """Build a single (annotation, Field) pair from a schema field config."""
    ftype = cfg["type"]
    desc = cfg.get("description", "")

    if ftype == "enum":
        values = cfg.get("values", [])
        if values:
            enum_cls = enum.Enum(
                name + "_enum",
                {v.replace(" ", "_").lower(): v for v in values},
                type=str,
            )
            return enum_cls | None, Field(None, description=desc)
        infer_desc = (
            desc + " (infer possible values from the data)"
            if desc
            else "Infer possible values from the data"
        )
        return str | None, Field(None, description=infer_desc)
    elif ftype == "number":
        return float | None, Field(None, description=desc)
    elif ftype == "array":
        sub_fields = {}
        for sub_name, sub_cfg in cfg.get("fields", {}).items():
            sub_fields[sub_name] = _build_field(sub_name, sub_cfg)
        sub_model = create_model(name + "_item", **sub_fields)
        return list[sub_model] | None, Field(None, description=desc)
    else:  # string, date
        return str | None, Field(None, description=desc)


def _build_model(spec: dict) -> type[BaseModel]:
    """Build a Pydantic model from a schema spec dict."""
    field_definitions = {}
    for name, cfg in spec["fields"].items():
        field_definitions[name] = _build_field(name, cfg)

    model = create_model(
        spec.get("name", "ExtractedData").replace(" ", ""),
        **field_definitions,
    )

    if spec.get("record_type") == "array":
        model = create_model(
            spec.get("name", "ExtractedData").replace(" ", "") + "List",
            items=(
                list[model],
                Field(..., description="List of extracted records"),
            ),
        )

    return model


def load_schema(
    schema_path: str | Path,
) -> tuple[type[BaseModel], dict]:
    """Load a YAML schema and return (PydanticModel, schema_meta)."""
    with open(schema_path) as f:
        spec = yaml.safe_load(f)
    return _build_model(spec), spec


def load_schema_from_dict(spec: dict) -> tuple[type[BaseModel], dict]:
    """Build a Pydantic model from an in-memory schema dict."""
    return _build_model(spec), spec


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n\n".join(page.get_text("text") for page in doc)


def _make_messages(text: str, instructions: str = "") -> list[dict]:
    system = SYSTEM
    if instructions:
        system += "\n\nAdditional instructions:\n" + instructions
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Extract fields from this document:\n\n{text}",
        },
    ]


def extract(
    pdf_path: str,
    response_model: type[BaseModel],
    model: str = "gpt-4.1-mini",
    instructions: str = "",
) -> BaseModel:
    text = extract_text(pdf_path)
    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=2,
        messages=_make_messages(text, instructions),
        temperature=0,
    )


async def async_extract(
    pdf_path: str,
    response_model: type[BaseModel],
    model: str = "gpt-4.1-mini",
    instructions: str = "",
) -> BaseModel:
    text = extract_text(pdf_path)
    return await async_client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=2,
        messages=_make_messages(text, instructions),
        temperature=0,
    )


def list_schemas() -> list[dict]:
    """Return available schemas from the schemas directory."""
    schemas = []
    for p in sorted(SCHEMAS_DIR.glob("*.yaml")):
        with open(p) as f:
            spec = yaml.safe_load(f)
        schemas.append({
            "file": p.name,
            "name": spec.get("name", p.stem),
            "description": spec.get("description", ""),
            "fields": list(spec.get("fields", {}).keys()),
        })
    return schemas


if __name__ == "__main__":
    import sys

    schema_path = SCHEMAS_DIR / "par_decision.yaml"
    response_model, spec = load_schema(schema_path)

    par_dir = BASE_DIR / "PAR_files"
    if len(sys.argv) > 1:
        files = [par_dir / f for f in sys.argv[1:]]
    else:
        files = sorted(par_dir.glob("*.pdf"))[:3]

    print(f"Schema: {spec['name']}")
    print(f"Fields: {list(spec['fields'].keys())}\n")

    for pdf_path in files:
        print(f"{'='*60}")
        print(f"FILE: {pdf_path.name}")
        print(f"{'='*60}")
        result = extract(str(pdf_path), response_model)
        print(result.model_dump_json(indent=2))
        print()
