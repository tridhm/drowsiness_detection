from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Config key '{key}' expects a boolean value.")


def _preferred_option(action: argparse.Action) -> str:
    for opt in action.option_strings:
        if opt.startswith("--"):
            return opt
    return action.option_strings[0]


def _load_json(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON root must be an object.")
    return data


def _default_schema_path() -> Path:
    return Path(__file__).with_name("config_tools.schema.json")


def _load_schema(path: str | None = None) -> dict[str, Any]:
    schema_path = Path(path) if path else _default_schema_path()
    if not schema_path.exists():
        raise FileNotFoundError(f"Config schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config schema root must be an object.")
    sections = data.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("Config schema must contain an object key: 'sections'.")
    return data


def _select_section(
    root: dict[str, Any],
    section: str | None,
    known_keys: set[str],
) -> dict[str, Any]:
    if section and section in root:
        section_data = root[section]
        if not isinstance(section_data, dict):
            raise ValueError(f"Config section '{section}' must be an object.")
        return section_data

    # Fallback: allow dedicated single-script config files.
    if any(key in known_keys for key in root.keys()):
        return root

    if section:
        print(f"[WARN] Config section '{section}' not found. Using script defaults.")
    return {}


def _get_section_schema(schema_root: dict[str, Any], section: str | None) -> dict[str, Any] | None:
    if section is None:
        return None
    sections = schema_root.get("sections", {})
    schema = sections.get(section)
    if schema is None:
        raise ValueError(f"Schema section '{section}' not found in config schema.")
    if not isinstance(schema, dict):
        raise ValueError(f"Schema section '{section}' must be an object.")
    return schema


def _validate_type(expected: str, value: Any, key: str) -> None:
    if expected == "string":
        if not isinstance(value, str):
            raise ValueError(f"Config key '{key}' expects type string, got {type(value).__name__}.")
        return

    if expected == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"Config key '{key}' expects type boolean, got {type(value).__name__}.")
        return

    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Config key '{key}' expects type integer, got {type(value).__name__}.")
        return

    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Config key '{key}' expects type number, got {type(value).__name__}.")
        return

    raise ValueError(f"Unsupported schema type '{expected}' for key '{key}'.")


def _validate_numeric_constraints(value: float, spec: dict[str, Any], key: str) -> None:
    minimum = spec.get("minimum")
    if minimum is not None and value < minimum:
        raise ValueError(f"Config key '{key}' must be >= {minimum}, got {value}.")

    maximum = spec.get("maximum")
    if maximum is not None and value > maximum:
        raise ValueError(f"Config key '{key}' must be <= {maximum}, got {value}.")

    exclusive_min = spec.get("exclusiveMinimum")
    if exclusive_min is not None and value <= exclusive_min:
        raise ValueError(f"Config key '{key}' must be > {exclusive_min}, got {value}.")

    exclusive_max = spec.get("exclusiveMaximum")
    if exclusive_max is not None and value >= exclusive_max:
        raise ValueError(f"Config key '{key}' must be < {exclusive_max}, got {value}.")


def _validate_string_constraints(value: str, spec: dict[str, Any], key: str) -> None:
    min_length = spec.get("minLength")
    if min_length is not None and len(value) < int(min_length):
        raise ValueError(f"Config key '{key}' length must be >= {min_length}, got {len(value)}.")

    max_length = spec.get("maxLength")
    if max_length is not None and len(value) > int(max_length):
        raise ValueError(f"Config key '{key}' length must be <= {max_length}, got {len(value)}.")


def _validate_payload(
    payload: dict[str, Any],
    section_schema: dict[str, Any] | None,
    section_name: str | None,
    known_keys: set[str],
) -> None:
    if not payload:
        return

    if section_schema is None:
        # If no section schema is requested, at least enforce parser-supported keys.
        unknown = sorted(key for key in payload.keys() if key not in known_keys)
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(unknown)}.")
        return

    fields = section_schema.get("fields")
    if not isinstance(fields, dict):
        raise ValueError(f"Schema section '{section_name}' must include an object key 'fields'.")
    allow_unknown = bool(section_schema.get("allow_unknown_keys", False))

    unknown: list[str] = []
    for key in payload.keys():
        if key not in fields:
            unknown.append(key)
        elif key not in known_keys:
            unknown.append(key)
    if unknown and not allow_unknown:
        raise ValueError(
            f"Unknown config key(s) in section '{section_name}': {', '.join(sorted(set(unknown)))}."
        )

    for key, value in payload.items():
        spec = fields.get(key)
        if not isinstance(spec, dict):
            continue

        expected_type = spec.get("type")
        if not isinstance(expected_type, str):
            raise ValueError(f"Schema for key '{key}' must define string field 'type'.")
        _validate_type(expected_type, value, key)

        enum_values = spec.get("enum")
        if enum_values is not None:
            if not isinstance(enum_values, list):
                raise ValueError(f"Schema enum for key '{key}' must be a list.")
            if value not in enum_values:
                raise ValueError(f"Config key '{key}' must be one of {enum_values}, got {value}.")

        if expected_type in {"integer", "number"}:
            _validate_numeric_constraints(float(value), spec, key)
        if expected_type == "string":
            _validate_string_constraints(value, spec, key)


def _build_config_tokens(
    parser: argparse.ArgumentParser,
    payload: dict[str, Any],
) -> list[str]:
    action_by_dest: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.dest == "help":
            continue
        action_by_dest[action.dest] = action

    tokens: list[str] = []

    for key, value in payload.items():
        action = action_by_dest.get(key)
        if action is None or key == "config":
            continue

        flag = _preferred_option(action)
        action_type = type(action).__name__

        if action_type == "_StoreTrueAction":
            if _parse_bool(value, key):
                tokens.append(flag)
            continue

        if action_type == "_StoreFalseAction":
            if not _parse_bool(value, key):
                tokens.append(flag)
            continue

        if value is None:
            continue

        if isinstance(value, list):
            tokens.append(flag)
            tokens.extend(str(item) for item in value)
        else:
            tokens.extend([flag, str(value)])
    return tokens


def parse_args_with_json_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
    section: str | None = None,
    schema_path: str | None = None,
) -> argparse.Namespace:
    cli_tokens = list(sys.argv[1:] if argv is None else argv)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args(cli_tokens)

    config_tokens: list[str] = []
    if pre_args.config:
        try:
            root = _load_json(pre_args.config)
            schema_root = _load_schema(schema_path)
            known_keys = {
                action.dest
                for action in parser._actions
                if action.dest not in {"help"} and action.option_strings
            }
            payload = _select_section(root, section, known_keys)
            section_schema = _get_section_schema(schema_root, section)
            _validate_payload(payload, section_schema, section, known_keys)
            config_tokens = _build_config_tokens(parser, payload)
        except Exception as exc:
            parser.error(str(exc))

    return parser.parse_args(config_tokens + cli_tokens)
