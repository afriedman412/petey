"""
Schema loading and Pydantic model building.
"""
import enum
import re

import yaml
from pathlib import Path
from pydantic import BaseModel, Field, create_model


def _safe_name(name: str) -> str:
    """Sanitize to match OpenAI's function name pattern: ^[a-zA-Z0-9_-]+$"""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _build_field(name: str, cfg: dict) -> tuple:
    ftype = cfg["type"]
    desc = cfg.get("description", "")

    if ftype in ("category", "enum"):
        values = cfg.get("values", [])
        if values:
            enum_cls = enum.Enum(
                _safe_name(name) + "_enum",
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
        sub_model = create_model(_safe_name(name) + "_item", **sub_fields)
        return list[sub_model] | None, Field(None, description=desc)
    else:  # string, date
        return str | None, Field(None, description=desc)


def build_model(spec: dict) -> type[BaseModel]:
    """Build a Pydantic model from a schema spec dict."""
    field_definitions = {}
    for name, cfg in spec["fields"].items():
        field_definitions[name] = _build_field(name, cfg)

    model_name = _safe_name(spec.get("name", "ExtractedData"))
    model = create_model(
        model_name,
        **field_definitions,
    )

    if spec.get("mode") == "table" or spec.get("record_type") == "array":
        model = create_model(
            model_name + "List",
            items=(
                list[model],
                Field(..., description="List of extracted records"),
            ),
        )

    return model


def load_schema(schema_path: str | Path) -> tuple[type[BaseModel], dict]:
    """Load a YAML schema file and return (PydanticModel, spec_dict)."""
    with open(schema_path) as f:
        spec = yaml.safe_load(f)
    return build_model(spec), spec
