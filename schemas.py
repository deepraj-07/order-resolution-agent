"""Declarative tool schemas, defined exactly once.

Both consumers share these plain-dict schemas:

* :func:`build_gemini_tools` converts them into ``google.genai.types.Tool``
  function declarations for the Gemini API.
* ``registry.dispatch`` validates model-provided arguments against the same
  dicts, so there is no drift between what the model sees and what is enforced.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Plain-dict schemas (single source of truth)
# ---------------------------------------------------------------------------

LOOKUP_ORDER_SCHEMA: Dict[str, Any] = {
    "name": "lookup_order",
    "description": "Get order facts and exchange eligibility. Call first.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Order ID.",
            },
        },
        "required": ["order_id"],
    },
}

CHECK_INVENTORY_SCHEMA: Dict[str, Any] = {
    "name": "check_inventory",
    "description": "Get replacement stock for an item.",
    "parameters": {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "Item ID.",
            },
        },
        "required": ["item_id"],
    },
}

PROCESS_REFUND_SCHEMA: Dict[str, Any] = {
    "name": "process_refund",
    "description": "Refund after lookup when exchange is impossible.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Exact order identifier to refund, e.g. 'ORD-1002'.",
            },
            "reason": {
                "type": "string",
                "description": "Short free-text reason for the refund, e.g. 'Damaged item, no replacement stock available'.",
            },
        },
        "required": ["order_id", "reason"],
    },
}

CONFIRM_EXCHANGE_SCHEMA: Dict[str, Any] = {
    "name": "confirm_exchange",
    "description": "Confirm an eligible, in-stock replacement.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Exact order identifier to exchange, e.g. 'ORD-1001'.",
            },
        },
        "required": ["order_id"],
    },
}

SEND_NOTIFICATION_SCHEMA: Dict[str, Any] = {
    "name": "send_notification",
    "description": "Email the completed resolution.",
    "parameters": {
        "type": "object",
        "properties": {
            "customer_email": {
                "type": "string",
                "description": "Customer email address from the order record, e.g. 'ana.silva@example.com'.",
            },
            "message": {
                "type": "string",
                "description": "Polite summary of the resolution to send to the customer.",
            },
        },
        "required": ["customer_email", "message"],
    },
}

#: Ordered list of all tool schemas (single source of truth).
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    LOOKUP_ORDER_SCHEMA,
    CHECK_INVENTORY_SCHEMA,
    PROCESS_REFUND_SCHEMA,
    CONFIRM_EXCHANGE_SCHEMA,
    SEND_NOTIFICATION_SCHEMA,
]

#: Map of tool name -> schema, for registry validation.
SCHEMAS_BY_NAME: Dict[str, Dict[str, Any]] = {schema["name"]: schema for schema in TOOL_SCHEMAS}


# ---------------------------------------------------------------------------
# Gemini conversion
# ---------------------------------------------------------------------------

def build_gemini_tools(names: List[str] | None = None) -> List[Any]:
    """Convert the dict schemas into ``google.genai.types.Tool`` declarations.

    The google-genai SDK is imported lazily so the offline MockLLM path keeps
    working even if the SDK is not installed.
    """
    from google.genai import types  # local import: offline-safe

    declarations = []
    for schema in TOOL_SCHEMAS:
        if names is not None and schema["name"] not in names:
            continue
        properties = {
            prop_name: types.Schema(
                type=getattr(types.Type, prop_def["type"].upper()),
                description=prop_def["description"],
            )
            for prop_name, prop_def in schema["parameters"]["properties"].items()
        }
        declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=schema["parameters"]["required"],
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)]
