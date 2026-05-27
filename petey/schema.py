"""
Blueprint loading and Pydantic model building.

The module name is retained as ``schema.py`` for backwards compatibility; the
public concept is "blueprint". Old names (``load_schema``, etc.) are kept as
deprecated wrappers and will be removed in v0.6.0.
"""
import enum
import re
import warnings

import yaml
from pathlib import Path
from typing import Annotated
from pydantic import BaseModel, BeforeValidator, Field, create_model


_DEPRECATION_TAIL = "Support will be removed in v0.6.0."


def _safe_name(name: str) -> str:
    """Sanitize to match OpenAI's function name pattern: ^[a-zA-Z0-9_-]+$"""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _safe_field_name(name: str) -> str:
    """Sanitize field name to match API property key patterns."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _build_field(name: str, cfg: dict) -> tuple:
    ftype = cfg["type"]
    desc = cfg.get("description", "")
    safe = _safe_field_name(name)
    alias = name if safe != name else None

    def field(**kw):
        if alias:
            return Field(
                serialization_alias=alias,
                **kw,
            )
        return Field(**kw)

    if ftype in ("category", "enum"):
        values = cfg.get("values", [])
        if values:
            enum_cls = enum.Enum(
                _safe_name(name) + "_enum",
                {v.replace(" ", "_").lower(): v for v in values},
                type=str,
            )
            # Case-insensitive enum matching
            _val_map = {v.lower(): v for v in values}

            def _coerce_enum(v, _map=_val_map):
                if isinstance(v, str):
                    match = _map.get(v.strip().lower())
                    if match is not None:
                        return match
                return v

            return (
                Annotated[enum_cls, BeforeValidator(_coerce_enum)] | None,
                field(default=None, description=desc),
            )
        infer_desc = (
            desc + " (infer possible values from the data)"
            if desc
            else "Infer possible values from the data"
        )
        return (
            str | None,
            field(default=None, description=infer_desc),
        )
    elif ftype == "number":
        return (
            float | None,
            field(default=None, description=desc),
        )
    elif ftype == "array":
        sub_fields = {}
        for sub_name, sub_cfg in cfg.get("fields", {}).items():
            s_safe = _safe_field_name(sub_name)
            sub_fields[s_safe] = _build_field(sub_name, sub_cfg)
        sub_model = create_model(
            _safe_name(name) + "_item", **sub_fields,
        )
        return (
            list[sub_model] | None,
            field(default=None, description=desc),
        )
    else:  # string, date
        return (
            str | None,
            field(default=None, description=desc),
        )


def build_model(spec: dict) -> type[BaseModel]:
    """Build a Pydantic model from a blueprint spec dict."""
    field_definitions = {}
    for name, cfg in spec["fields"].items():
        safe = _safe_field_name(name)
        field_definitions[safe] = _build_field(name, cfg)

    model_name = _safe_name(spec.get("name", "ExtractedData"))
    model = create_model(
        model_name,
        **field_definitions,
    )
    model.model_config["populate_by_name"] = True

    if spec.get("mode") == "table" or spec.get("record_type") == "array":
        model = create_model(
            model_name + "List",
            items=(
                list[model],
                Field(..., description="List of extracted records"),
            ),
        )

    return model


def load_blueprint(
    blueprint_path: str | Path,
) -> tuple[type[BaseModel], dict]:
    """Load a blueprint file and return (PydanticModel, spec_dict).

    Accepts both ``.bpt`` (canonical) and ``.yaml`` (deprecated) extensions.
    Loading a ``.yaml`` file emits a ``DeprecationWarning``.
    """
    path = Path(blueprint_path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        warnings.warn(
            f"The .yaml/.yml blueprint extension is deprecated; "
            f"rename to .bpt. {_DEPRECATION_TAIL}",
            DeprecationWarning,
            stacklevel=2,
        )
    with open(path) as f:
        spec = yaml.safe_load(f)
    return build_model(spec), spec


def load_schema(
    schema_path: str | Path,
) -> tuple[type[BaseModel], dict]:
    """(Deprecated) Use :func:`load_blueprint`. Removed in v0.6.0."""
    warnings.warn(
        f"load_schema() is deprecated; use load_blueprint(). "
        f"{_DEPRECATION_TAIL}",
        DeprecationWarning,
        stacklevel=2,
    )
    # Don't call load_blueprint to avoid stacking warnings; inline the work.
    path = Path(schema_path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        # Suppress the extension warning here — caller already got the
        # load_schema deprecation, no need to double-warn.
        pass
    with open(path) as f:
        spec = yaml.safe_load(f)
    return build_model(spec), spec


def _date_fields(spec: dict) -> set[str]:
    """Return the set of field names with type 'date' in *spec*."""
    names: set[str] = set()
    for name, cfg in spec.get("fields", {}).items():
        if cfg.get("type") == "date":
            names.add(_safe_field_name(name))
    return names


def normalize_dates(record: dict, spec: dict) -> dict:
    """Normalize date fields in *record* to YYYY-MM-DD format.

    Uses ``dateutil.parser`` to handle formats like
    "December 8, 1986", "1986-12-08", "DEC 30 1993", etc.
    Values that cannot be parsed are left unchanged.
    """
    from dateutil import parser as _dp

    fields = _date_fields(spec)
    if not fields:
        return record
    for key in fields:
        val = record.get(key)
        if not val or not isinstance(val, str):
            continue
        try:
            record[key] = _dp.parse(val).strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            pass  # leave unparseable values as-is
    return record
