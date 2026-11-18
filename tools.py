"""Deterministic business tools for the order resolution agent.

Every tool returns a plain dict with a ``"status"`` key of either
``"success"`` or ``"error"`` and never raises: failures are expressed as an
``error_code`` payload so the agent can feed them back to the model for
self-correction. All data lives in :mod:`database`.
"""

from __future__ import annotations

import re
from typing import Any, Dict

import database

# Error codes used across tools and the registry.
ORDER_NOT_FOUND = "ORDER_NOT_FOUND"
ITEM_NOT_FOUND = "ITEM_NOT_FOUND"
ALREADY_REFUNDED = "ALREADY_REFUNDED"
EXCHANGE_NOT_ELIGIBLE = "EXCHANGE_NOT_ELIGIBLE"
OUT_OF_STOCK = "OUT_OF_STOCK"
INVALID_EMAIL = "INVALID_EMAIL"

_EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_tool_error = lambda code, message: {  # noqa: E731 - tiny local helper
    "status": "error",
    "error_code": code,
    "message": message,
}


# ---------------------------------------------------------------------------
# 1. lookup_order
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> Dict[str, Any]:
    """Return the customer-facing details of an order.

    Returns ``error_code=ORDER_NOT_FOUND`` when the order id is unknown.
    """
    key = database.normalize_id(order_id)
    order = database.get_order(key)
    if order is None:
        return _tool_error(
            ORDER_NOT_FOUND,
            f"Order '{key}' was not found in the system. Ask the customer to "
            "verify the order ID.",
        )
    return {
        "status": "success",
        "order_id": order["order_id"],
        "customer_name": order["customer_name"],
        "customer_email": order["customer_email"],
        "item_id": order["item_id"],
        "item_name": order["item_name"],
        "price": order["price"],
        "order_status": order["order_status"],
        "exchange_eligible": bool(order["exchange_eligible"]),
        "eligibility_reason": order["eligibility_reason"],
        "payment_method": order["payment_method"],
    }


# ---------------------------------------------------------------------------
# 2. check_inventory
# ---------------------------------------------------------------------------

def check_inventory(item_id: str) -> Dict[str, Any]:
    """Return current stock for an item.

    Returns ``error_code=ITEM_NOT_FOUND`` when the item id is unknown.
    """
    key = database.normalize_id(item_id)
    record = database.get_inventory(key)
    if record is None:
        return _tool_error(
            ITEM_NOT_FOUND,
            f"Item '{key}' is not in the inventory catalogue. "
            "Verify the item_id from the order record.",
        )
    stock = int(record["stock_count"])
    return {
        "status": "success",
        "item_id": key,
        "stock_count": stock,
        "available": stock > 0,
    }


# ---------------------------------------------------------------------------
# 3. process_refund
# ---------------------------------------------------------------------------

def process_refund(order_id: str, reason: str) -> Dict[str, Any]:
    """Issue a financial refund to the customer's original payment method.

    Guards against double refunds with ``error_code=ALREADY_REFUNDED``.
    """
    key = database.normalize_id(order_id)
    order = database.get_order(key)
    if order is None:
        return _tool_error(
            ORDER_NOT_FOUND,
            f"Order '{key}' was not found. Do not attempt to refund a order "
            "that does not exist; ask the customer to verify the order ID.",
        )

    existing = database.get_refund_for_order(key)
    if existing is not None:
        return _tool_error(
            ALREADY_REFUNDED,
            f"Order '{key}' was already refunded under reference "
            f"'{existing['refund_reference']}' for "
            f"{existing['refunded_amount']:.2f} USD on "
            f"{existing['created_at']}. Do not refund again.",
        )

    refund = database.add_refund(
        order_id=key,
        amount=float(order["price"]),
        refunded_to=order["payment_method"],
        reason=reason,
    )
    database.set_order_status(key, "refunded")
    return {
        "status": "success",
        "order_id": key,
        "refund_reference": refund["refund_reference"],
        "refunded_amount": refund["refunded_amount"],
        "refunded_to": refund["refunded_to"],
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 4. confirm_exchange
# ---------------------------------------------------------------------------

def confirm_exchange(order_id: str) -> Dict[str, Any]:
    """Confirm a replacement shipment for an eligible order with stock.

    - ``EXCHANGE_NOT_ELIGIBLE`` when the order is not exchange eligible.
    - ``OUT_OF_STOCK`` when the replacement item has no stock left.
    """
    key = database.normalize_id(order_id)
    order = database.get_order(key)
    if order is None:
        return _tool_error(
            ORDER_NOT_FOUND,
            f"Order '{key}' was not found. Ask the customer to verify the "
            "order ID.",
        )
    if not order.get("exchange_eligible", False):
        return _tool_error(
            EXCHANGE_NOT_ELIGIBLE,
            f"Order '{key}' is not eligible for exchange: "
            f"{order.get('eligibility_reason', 'unknown reason')} "
            "A refund may be appropriate instead.",
        )
    if order.get("order_status") == "refunded":
        return _tool_error(
            EXCHANGE_NOT_ELIGIBLE,
            f"Order '{key}' was already refunded; an exchange is no longer "
            "possible. Apologize and explain the refund stands.",
        )

    stock_record = database.get_inventory(order["item_id"])
    if stock_record is None:
        return _tool_error(ITEM_NOT_FOUND, f"Item '{order['item_id']}' is not in the catalogue.")
    if int(stock_record["stock_count"]) <= 0:
        return _tool_error(
            OUT_OF_STOCK,
            f"'{order['item_name']}' is out of stock (0 units). A refund is "
            "the appropriate resolution.",
        )

    remaining = database.decrement_stock(order["item_id"])
    exchange = database.add_exchange(key)
    database.set_order_status(key, "exchange_confirmed")
    return {
        "status": "success",
        "order_id": key,
        "exchange_reference": exchange["exchange_reference"],
        "item_id": order["item_id"],
        "item_name": order["item_name"],
        "remaining_stock": remaining,
    }


# ---------------------------------------------------------------------------
# 5. send_notification
# ---------------------------------------------------------------------------

def send_notification(customer_email: str, message: str) -> Dict[str, Any]:
    """Deliver a notification to the customer's email address.

    Validates the email format and logs the notification in the database.
    """
    email = customer_email.strip()
    if not _EMAIL_REGEX.match(email):
        return _tool_error(
            INVALID_EMAIL,
            f"'{customer_email}' is not a valid email address. Use the "
            "customer_email returned by lookup_order.",
        )
    notification = database.add_notification(email=email, message=message)
    return {
        "status": "success",
        "notification_id": notification["notification_id"],
        "customer_email": email,
        "delivered": True,
        "message": message,
    }
