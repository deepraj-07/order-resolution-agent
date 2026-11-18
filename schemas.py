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
    "description": (
        "Look up an order by its exact order ID. Returns the customer name "
        "and email, the ordered item, the price, the order status, whether "
        "the order is exchange-eligible (with the reason), and the original "
        "payment method. Always call this first, before any other tool, so "
        "you know if the order exists and what its eligibility is."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Exact order identifier, e.g. 'ORD-1001'.",
            },
        },
        "required": ["order_id"],
    },
}

CHECK_INVENTORY_SCHEMA: Dict[str, Any] = {
    "name": "check_inventory",
    "description": (
        "Check the warehouse stock for an item ID. Returns the stock count "
        "and whether a replacement unit is available. Call this after "
        "lookup_order using the item_id from the order record."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "item_id": {
                "type": "string",
                "description": "Exact item identifier from the order record, e.g. 'ITEM-LAPTOP-15'.",
            },
        },
        "required": ["item_id"],
    },
}

PROCESS_REFUND_SCHEMA: Dict[str, Any] = {
    "name": "process_refund",
    "description": (
        "Issue a financial refund to the customer's original payment method. "
        "Use ONLY when a replacement is unavailable (stock is 0) or the order "
        "is not eligible for exchange. Never use when the exchange is "
        "possible. Never refund an order you have not looked up, and never "
        "refund twice."
    ),
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
    "description": (
        "Confirm a replacement shipment. Use ONLY when the order is "
        "exchange-eligible AND stock > 0. Decrement the item stock by one "
        "and lock in the replacement."
    ),
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
    "description": (
        "Send a confirmation notification to the customer by email. Use the "
        "customer_email from lookup_order. Wrap up every successful "
        "resolution (refund or exchange) with exactly one polite "
        "notification describing what was done."
    ),
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

def build_gemini_tools() -> List[Any]:
    """Convert the dict schemas into ``google.genai.types.Tool`` declarations.

    The google-genai SDK is imported lazily so the offline MockLLM path keeps
    working even if the SDK is not installed.
    """
    from google.genai import types  # local import: offline-safe

    declarations = []
    for schema in TOOL_SCHEMAS:
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
