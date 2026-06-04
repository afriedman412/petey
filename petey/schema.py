"""
Blueprint loading and Pydantic model building.

Reference implementation of the BPT Spec v1.0 (see ``BPT SPEC v1.md`` at the
repo root). Loads a ``.bpt`` blueprint document, validates it against the
spec, and returns a Pydantic model representing the extraction target schema.

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


# BPT Spec v1.0 §5.3: type-name aliases. All map to the canonical form.
_TYPE_ALIASES = {
    "bool": "boolean",
    "enum": "category",
    "cat": "category",
}


def _normalize_type(ftype: str | None) -> str | None:
    """Map an alias type name to its canonical form (BPT Spec v1.0 §5.3)."""
    if ftype is None:
        return None
    return _TYPE_ALIASES.get(ftype, ftype)


def _safe_name(name: str) -> str:
    """Sanitize to match OpenAI's function name pattern: ^[a-zA-Z0-9_-]+$"""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _safe_field_name(name: str) -> str:
    """Sanitize field name to match API property key patterns."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _build_field(name: str, cfg: dict) -> tuple:
    ftype = _normalize_type(cfg["type"])
    desc = cfg.get("description", "")
    pattern = cfg.get("pattern")
    safe = _safe_field_name(name)
    alias = name if safe != name else None

    def field(**kw):
        if alias:
            return Field(
                serialization_alias=alias,
                **kw,
            )
        return Field(**kw)

    if ftype == "category":
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
    elif ftype == "boolean":
        # BPT Spec v1.0 §5.3.5. Pydantic coerces 1/0, "1"/"0", "true"/"false",
        # "yes"/"no" by default.
        return (
            bool | None,
            field(default=None, description=desc),
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
        if ftype == "string" and pattern is not None:
            # §5.6 — pattern is an output-shape signal for the LLM.
            # Surfacing it in the field description gets it into the
            # JSON schema instructor sends to the model. We don't
            # validate extracted values against it; pattern shapes
            # output, it doesn't gate input. Patterns declared on
            # non-string fields are silently ignored.
            augmented_desc = (
                f"{desc} Must match the canonical regex pattern: "
                f"{pattern}"
                if desc
                else f"Must match the canonical regex pattern: {pattern}"
            )
            return (
                str | None,
                field(default=None, description=augmented_desc),
            )
        return (
            str | None,
            field(default=None, description=desc),
        )


def _validate_field_cfg(name: str, cfg: dict) -> None:
    """Recursively validate one field config against BPT Spec v1.0.

    Currently enforces §5.6 (pattern only on type: string; pattern must be a
    compilable regex). Walks into array children. Silent on malformed shapes
    that aren't dicts — build_model raises a clearer error there.
    """
    if not isinstance(cfg, dict):
        return

    ftype = _normalize_type(cfg.get("type"))
    pattern = cfg.get("pattern")
    if pattern is not None:
        if ftype != "string":
            raise ValueError(
                f"Blueprint field {name!r} declares `pattern` but its "
                f"type is {ftype!r}, not 'string' "
                f"(BPT Spec v1.0 §5.6)."
            )
        if not isinstance(pattern, str):
            raise ValueError(
                f"Blueprint field {name!r}: `pattern` must be a string, "
                f"got {type(pattern).__name__}."
            )
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(
                f"Blueprint field {name!r} has an invalid regex "
                f"`pattern` {pattern!r}: {e}."
            )

    if ftype == "array":
        for sub_name, sub_cfg in (cfg.get("fields") or {}).items():
            _validate_field_cfg(sub_name, sub_cfg)


def validate_blueprint(spec: dict) -> None:
    """Validate a blueprint spec against BPT Spec v1.0.

    Raises ``ValueError`` on any rule violation. This is the load-time
    spec-conformance check; ``load_blueprint`` calls it before
    ``build_model``. ``build_model`` itself is intentionally permissive
    (it silently skips spec-level violations) so callers that hand-build
    spec dicts in tests aren't forced through this gate.

    Currently checks:
    - §5.6: ``pattern`` is only valid on ``type: string`` fields; the
      value must be a compilable regex.
    """
    if not isinstance(spec, dict):
        return
    fields = spec.get("fields") or {}
    for name, cfg in fields.items():
        _validate_field_cfg(name, cfg)


def _resolve_parent_refs(fields: dict) -> dict:
    """Resolve parent-reference composition into inline form (BPT Spec v1.0 §5.5).

    Fields with a ``parent`` key are moved out of the top level and into the
    referenced array's ``fields`` mapping. The input is not mutated.

    Raises ``ValueError`` for the rule violations in §5.5.3:
    - parent name not defined at top level
    - parent is not type ``array``
    - parent already has an inline ``fields`` key (mixed modes)
    """
    has_parent_refs = any(
        isinstance(cfg, dict) and "parent" in cfg for cfg in fields.values()
    )
    if not has_parent_refs:
        return fields

    resolved = {}
    parent_groups: dict[str, dict] = {}

    for name, cfg in fields.items():
        if isinstance(cfg, dict) and "parent" in cfg:
            parent = cfg["parent"]
            child_cfg = {k: v for k, v in cfg.items() if k != "parent"}
            parent_groups.setdefault(parent, {})[name] = child_cfg
        else:
            # Defensive copy so we can mutate `fields` below without
            # touching the caller's dict.
            resolved[name] = dict(cfg) if isinstance(cfg, dict) else cfg

    for parent_name, children in parent_groups.items():
        if parent_name not in resolved:
            raise ValueError(
                f"Blueprint field {parent_name!r} is referenced as a parent "
                f"but is not defined at the top level of `fields` "
                f"(BPT Spec v1.0 §5.5.3 rule 1)."
            )
        parent_cfg = resolved[parent_name]
        if not isinstance(parent_cfg, dict):
            raise ValueError(
                f"Blueprint parent {parent_name!r} must be a field definition."
            )
        if _normalize_type(parent_cfg.get("type")) != "array":
            raise ValueError(
                f"Blueprint field {parent_name!r} is referenced as a parent "
                f"but its type is {parent_cfg.get('type')!r}, not 'array' "
                f"(BPT Spec v1.0 §5.5.3 rule 2)."
            )
        if parent_cfg.get("fields"):
            raise ValueError(
                f"Blueprint array {parent_name!r} declares both inline "
                f"`fields` and is referenced by `parent`. Use exactly one "
                f"composition mode (BPT Spec v1.0 §5.5.3 rule 3)."
            )
        parent_cfg["fields"] = children

    return resolved


def build_model(spec: dict) -> type[BaseModel]:
    """Build a Pydantic model from a blueprint spec dict.

    Implements BPT Spec v1.0. Resolves parent-reference composition (§5.5)
    before building the model. Field-level ``required`` flags (§5.4) are
    preserved in the spec dict for downstream consumers but do not affect
    the Pydantic model's nullability — every field remains nullable in
    the output type per §5.4. ``pattern`` (§5.6) on ``type: string`` fields
    is forwarded to the LLM via the field description to shape the canonical
    output form. Patterns on non-string fields are silently ignored;
    extracted values are not validated against the pattern (BPT runtime
    validation, when it lands, owns that check).
    """
    field_definitions = {}
    resolved_fields = _resolve_parent_refs(spec["fields"])
    for name, cfg in resolved_fields.items():
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
    validate_blueprint(spec)
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
    validate_blueprint(spec)
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
