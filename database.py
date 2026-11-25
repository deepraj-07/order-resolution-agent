"""In-memory mock database for the order resolution agent.

This module is the ONLY place in the project holding mutable global state.
All tool functions read and write through the accessors defined here, which
keeps side effects observable for the demo output and for tests.

The store is deterministic: :func:`reset_db` restores the exact seed data,
so each demo scenario and each test starts from a clean, known state.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Storage (the only mutable globals in the project)
# ---------------------------------------------------------------------------

_ORDERS: Dict[str, Dict[str, Any]] = {}
_INVENTORY: Dict[str, Dict[str, Any]] = {}
_REFUNDS: List[Dict[str, Any]] = []
_EXCHANGES: List[Dict[str, Any]] = []
_NOTIFICATIONS: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_ORDER_SEED: List[Dict[str, Any]] = [
    {
        "order_id": "ORD-1001",
        "customer_name": "Ana Silva",
        "customer_email": "ana.silva@example.com",
        "item_id": "ITEM-LAPTOP-15",
        "item_name": "Nimbus Pro 15 Laptop",
        "price": 1299.99,
        "order_status": "delivered",
        "exchange_eligible": True,
        "eligibility_reason": "Within the 30-day return window.",
        "payment_method": "Visa credit card ending 4242",
    },
    {
        "order_id": "ORD-1002",
        "customer_name": "Ben Okafor",
        "customer_email": "ben.okafor@example.com",
        "item_id": "ITEM-HEADPHONE",
        "item_name": "Pulse Studio Headphones",
        "price": 249.00,
        "order_status": "delivered",
        "exchange_eligible": True,
        "eligibility_reason": "Within the 30-day return window.",
        "payment_method": "PayPal (ben.okafor@example.com)",
    },
    {
        "order_id": "ORD-1003",
        "customer_name": "Carla Ruiz",
        "customer_email": "carla.ruiz@example.com",
        "item_id": "ITEM-KETTLE",
        "item_name": "AquaTemp Glass Kettle",
        "price": 59.95,
        "order_status": "delivered",
        "exchange_eligible": False,
        "eligibility_reason": "Final sale item; returns are not accepted.",
        "payment_method": "Mastercard debit card ending 8871",
    },
    {
        "order_id": "ORD-1004",
        "customer_name": "Dan Li",
        "customer_email": "dan.li@example.com",
        "item_id": "ITEM-CAMERA",
        "item_name": "Vista X3 Action Camera",
        "price": 499.00,
        "order_status": "refunded",
        "exchange_eligible": False,
        "eligibility_reason": "Order was already refunded on 2024-11-01; "
        "no further resolution applies.",
        "payment_method": "Amex card ending 3007",
    },
    {
        "order_id": "ORD-1005",
        "customer_name": "Elena Petrova",
        "customer_email": "elena.petrova@example.com",
        "item_id": "ITEM-KEYBOARD",
        "item_name": "TactileMax Mechanical Keyboard",
        "price": 149.50,
        "order_status": "delivered",
        "exchange_eligible": True,
        "eligibility_reason": "Within the 30-day return window.",
        "payment_method": "Visa credit card ending 1119",
    },
    {
        "order_id": "ORD-1006",
        "customer_name": "Frank Moreau",
        "customer_email": "frank.moreau@example.com",
        "item_id": "ITEM-CAMERA",
        "item_name": "Vista X3 Action Camera",
        "price": 499.00,
        "order_status": "delivered",
        "exchange_eligible": False,
        "eligibility_reason": "Outside the 30-day return window (order placed 62 days ago).",
        "payment_method": "Shop Pay (frank.moreau@example.com)",
    },
]

_INVENTORY_SEED: Dict[str, int] = {
    "ITEM-LAPTOP-15": 4,
    "ITEM-HEADPHONE": 0,
    "ITEM-KETTLE": 0,
    "ITEM-CAMERA": 2,
    "ITEM-KEYBOARD": 7,
}

_REFUND_SEED: List[Dict[str, Any]] = [
    {
        "order_id": "ORD-1004",
        "refund_reference": "REF-LEGACY42",
        "refunded_amount": 499.00,
        "refunded_to": "Amex card ending 3007",
        "reason": "Item lost in transit (prior support ticket #8841).",
        "created_at": "2024-11-01T14:32:00Z",
    }
]


def _seed() -> None:
    """Load deterministic seed data into every store."""
    _ORDERS.clear()
    _INVENTORY.clear()
    _REFUNDS.clear()
    _EXCHANGES.clear()
    _NOTIFICATIONS.clear()

    for order in _ORDER_SEED:
        _ORDERS[order["order_id"]] = dict(order)

    for item_id, stock in _INVENTORY_SEED.items():
        _INVENTORY[item_id] = {"item_id": item_id, "stock_count": stock}

    for refund in _REFUND_SEED:
        _REFUNDS.append(dict(refund))


# Seed once at import so the module is usable without an explicit reset.
_seed()


# ---------------------------------------------------------------------------
# Reference-id helpers
# ---------------------------------------------------------------------------

def _new_reference(prefix: str) -> str:
    """Generate a short reference such as ``REF-1A2B3C``."""
    return f"{prefix}-{secrets.token_hex(3).upper()}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def normalize_id(value: str) -> str:
    """Normalize an identifier: trim whitespace and uppercase."""
    return str(value).strip().upper()


def get_order(order_id: str) -> Dict[str, Any] | None:
    """Return a copy of the order, or None when unknown."""
    order = _ORDERS.get(normalize_id(order_id))
    return dict(order) if order else None


def set_order_status(order_id: str, status: str) -> None:
    """Update an order's status (used by refund/exchange tools)."""
    key = normalize_id(order_id)
    if key in _ORDERS:
        _ORDERS[key]["order_status"] = status


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def get_inventory(item_id: str) -> Dict[str, Any] | None:
    """Return a copy of the inventory record, or None when unknown."""
    item = _INVENTORY.get(normalize_id(item_id))
    return dict(item) if item else None


def decrement_stock(item_id: str) -> int:
    """Decrement stock by one and return the new count."""
    key = normalize_id(item_id)
    record = _INVENTORY[key]
    record["stock_count"] = max(0, record["stock_count"] - 1)
    return record["stock_count"]


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------

def get_refund_for_order(order_id: str) -> Dict[str, Any] | None:
    """Return the existing refund for an order, or None."""
    key = normalize_id(order_id)
    for refund in _REFUNDS:
        if refund["order_id"] == key:
            return dict(refund)
    return None


def add_refund(order_id: str, amount: float, refunded_to: str, reason: str) -> Dict[str, Any]:
    """Record a refund and return it."""
    refund = {
        "order_id": normalize_id(order_id),
        "refund_reference": _new_reference("REF"),
        "refunded_amount": amount,
        "refunded_to": refunded_to,
        "reason": reason,
        "created_at": _utc_now_iso(),
    }
    _REFUNDS.append(refund)
    return dict(refund)


def list_refunds() -> List[Dict[str, Any]]:
    """All refunds recorded so far (for demo side-effect reporting)."""
    return [dict(refund) for refund in _REFUNDS]


# ---------------------------------------------------------------------------
# Exchanges
# ---------------------------------------------------------------------------

def add_exchange(order_id: str) -> Dict[str, Any]:
    """Record a confirmed exchange and return it."""
    exchange = {
        "order_id": normalize_id(order_id),
        "exchange_reference": _new_reference("EXG"),
        "created_at": _utc_now_iso(),
    }
    _EXCHANGES.append(exchange)
    return exchange


def list_exchanges() -> List[Dict[str, Any]]:
    """All exchanges confirmed so far."""
    return [dict(exchange) for exchange in _EXCHANGES]


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def add_notification(email: str, message: str) -> Dict[str, Any]:
    """Log a delivery-confirmed notification."""
    notification = {
        "notification_id": _new_reference("NTF"),
        "customer_email": email,
        "message": message,
        "delivered": True,
        "created_at": _utc_now_iso(),
    }
    _NOTIFICATIONS.append(notification)
    return notification


def list_notifications() -> List[Dict[str, Any]]:
    """All notifications sent so far."""
    return [dict(notification) for notification in _NOTIFICATIONS]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset_db() -> None:
    """Restore the exact seed state (used between demo scenarios and tests)."""
    _seed()
    # The demo database has changed wholesale; cached read models are stale.
    # Local import avoids coupling the persistence layer to optional optimizers.
    try:
        from optimization import CACHE
        CACHE.clear()
    except ImportError:  # pragma: no cover - standalone DB use
        pass
