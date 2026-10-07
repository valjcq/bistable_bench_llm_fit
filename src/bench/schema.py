"""A tiny JSON Schema validator for the tool schemas (no extra dependency).

Supports exactly the keywords our tool schemas use: type (a name or a list of names),
properties, required, additionalProperties (false only), enum, minimum, maximum,
exclusiveMinimum, items, minItems, maxItems. Anything else in a schema is ignored, so keep the
schemas inside this subset (tests check the shipped ones). The schemas themselves are plain
JSON Schema, so an LLM backend or the full `jsonschema` package can consume them unchanged.
"""

from __future__ import annotations

from typing import Any

_TYPES: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
}


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of human-readable problems (empty when `value` satisfies `schema`)."""
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else types
        if not any(_TYPES[n](value) for n in names):
            return [f"{path}: expected {' or '.join(names)}, got {type(value).__name__}"]
    problems: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"{path}: must be one of {schema['enum']}, got {value!r}")
    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: must be >= {schema['minimum']}, got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: must be <= {schema['maximum']}, got {value}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: must be > {schema['exclusiveMinimum']}, got {value}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        problems += [f"{path}: missing required '{k}'" for k in schema.get("required", [])
                     if k not in value]  # fmt: skip
        for k, v in value.items():
            if k in props:
                problems += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                problems.append(f"{path}: unexpected property '{k}'")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: needs at most {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                problems += validate(v, schema["items"], f"{path}[{i}]")
    return problems
