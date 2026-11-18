"""Whitelist tool registry with a safe dispatcher.

The model can only influence the system through :func:`dispatch`, which:

1. rejects tool names not present in the whitelist (``UNAUTHORIZED_TOOL``),
2. validates arguments against the declarative schemas (``INVALID_ARGUMENTS``
   with a message precise enough for the model to self-correct), and
3. wraps execution so an unexpected bug returns ``TOOL_EXECUTION_ERROR``
   instead of crashing the agent loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict

from schemas import SCHEMAS_BY_NAME
import tools

logger = logging.getLogger(__name__)

# Error codes produced by the dispatcher itself.
UNAUTHORIZED_TOOL = "UNAUTHORIZED_TOOL"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
UNKNOWN_TOOL_KEY = "UNKNOWN_TOOL_KEY"

#: Whitelist registry: only these tool names can ever be executed.
TOOL_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "lookup_order": tools.lookup_order,
    "check_inventory": tools.check_inventory,
    "process_refund": tools.process_refund,
    "confirm_exchange": tools.confirm_exchange,
    "send_notification": tools.send_notification,
}


def _error(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "error_code": code, "message": message}


def _parse_raw_args(raw_args: Any) -> Dict[str, Any]:
    """Normalize raw model args (dict or JSON string) into a dict or an error."""
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return _error(
                INVALID_ARGUMENTS,
                f"Arguments were passed as an invalid JSON string ({exc.msg} at "
                f"position {exc.pos}). Send a valid JSON object of arguments.",
            )
    if not isinstance(raw_args, dict):
        return _error(
            INVALID_ARGUMENTS,
            f"Arguments must be a JSON object of name/value pairs, got "
            f"{type(raw_args).__name__} instead.",
        )
    return raw_args


def _validate_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any] | None:
    """Validate args against the schema.

    Returns None when valid; otherwise an ``INVALID_ARGUMENTS`` error dict
    that explains exactly what to fix so the model can self-correct.
    """
    schema = SCHEMAS_BY_NAME.get(tool_name)
    properties: Dict[str, Any] = schema["parameters"]["properties"]
    required: list[str] = schema["parameters"]["required"]

    problems: list[str] = []
    for field in required:
        if field not in args or args[field] is None:
            problems.append(f"missing required argument '{field}'")
    if problems:
        return _error(
            INVALID_ARGUMENTS,
            f"Tool '{tool_name}' cannot run: { '; '.join(problems) }. "
            f"Expected arguments: {sorted(properties)}.",
        )

    for field, value in args.items():
        if field not in properties:
            problems.append(f"unknown argument '{field}'")
            continue
        expected = properties[field]["type"].lower()
        if expected == "string" and (not isinstance(value, str) or not value.strip()):
            problems.append(
                f"argument '{field}' must be a non-empty string, got "
                f"{type(value).__name__} ({value!r})"
            )
        # Extend here for other primitive types if new tools need them.
    if problems:
        expected_args = ", ".join(
            "{} ({})".format(name, prop["type"]) for name, prop in properties.items()
        )
        return _error(
            INVALID_ARGUMENTS,
            f"Tool '{tool_name}' cannot run: { '; '.join(problems) }. "
            f"Fix the arguments and call the tool again. Expected arguments: "
            f"{expected_args}.",
        )
    return None


def dispatch(tool_name: str, raw_args: Any) -> Dict[str, Any]:
    """Safely execute a whitelisted tool. Never raises.

    ``raw_args`` may be a dict (as returned by Gemini) or a JSON string
    produced directly by a model's text output.
    """
    if tool_name not in TOOL_REGISTRY:
        logger.warning("Blocked unauthorized tool call: %r", tool_name)
        return _error(
            UNAUTHORIZED_TOOL,
            f"Tool '{tool_name}' is not permitted. Allowed tools: "
            f"{sorted(TOOL_REGISTRY)}.",
        )

    args = _parse_raw_args(raw_args)
    if "status" in args and "error_code" in args:
        return args  # already an INVALID_ARGUMENTS payload

    validation = _validate_args(tool_name, args)
    if validation is not None:
        logger.warning("Invalid arguments for %s: %s", tool_name, validation["message"])
        return validation

    try:
        output = TOOL_REGISTRY[tool_name](**args)
    except Exception as exc:  # noqa: BLE001 - the loop must never crash
        logger.exception("Tool '%s' raised while executing", tool_name)
        return _error(
            TOOL_EXECUTION_ERROR,
            f"Tool '{tool_name}' failed with an unexpected error: {exc}. "
            "You may retry with corrected arguments.",
        )

    if not isinstance(output, dict):
        return _error(
            TOOL_EXECUTION_ERROR,
            f"Tool '{tool_name}' returned a non-dict result ({type(output).__name__}).",
        )
    logger.debug("dispatch(%s) -> %s", tool_name, output.get("status", "?"))
    return output
