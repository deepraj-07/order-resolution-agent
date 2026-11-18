"""Pytest suite covering every acceptance criterion (offline, via MockLLM).

No network access is used: the suite exercises the agent loop, guardrails,
registry security and tool semantics through the rule-based MockLLM and by
calling tools directly.
"""

from __future__ import annotations

import json

import pytest

import database
from agent import DEFAULT_MAX_ITERATIONS, ESCALATION_MESSAGE, ResolutionAgent
from llm import MockLLM
from registry import TOOL_REGISTRY, dispatch


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_db():
    """Every test starts and ends with pristine seed data."""
    database.reset_db()
    yield
    database.reset_db()


@pytest.fixture()
def agent():
    return ResolutionAgent(llm=MockLLM())


#: Refunds that exist purely from seed data (ORD-1004 was refunded before).
SEED_REFUND_REFS = {"REF-LEGACY42"}


def tool_sequence(result) -> list[str]:
    """Flat ordered list of tool names from the trace."""
    return [call["name"] for entry in result.trace for call in entry["action"]]


def run(agent, message: str):
    return agent.run(message)


# ---------------------------------------------------------------------------
# Autonomy and decision logic
# ---------------------------------------------------------------------------

def test_autonomous_investigation_lookup_and_inventory_without_hints(agent):
    """The agent must look up the order and check stock on its own."""
    message = "My laptop from ORD-1001 arrived damaged."
    result = run(agent, message)

    assert "lookup_order" in tool_sequence(result)
    assert "check_inventory" in tool_sequence(result)
    # No tool name is mentioned in the user's words.
    for tool_name in tool_sequence(result):
        assert tool_name not in message


def test_stock_zero_triggers_refund_not_exchange(agent):
    result = run(
        agent, "The headphones in order ORD-1002 arrived damaged. Help!"
    )
    seq = tool_sequence(result)
    assert "process_refund" in seq
    assert "confirm_exchange" not in seq
    assert not result.escalated


def test_stock_available_and_eligible_triggers_exchange_not_refund(agent):
    result = run(agent, "My order ORD-1001 arrived damaged, please help.")
    seq = tool_sequence(result)
    assert "confirm_exchange" in seq
    assert "process_refund" not in seq
    # Side effect: stock decremented by exactly one.
    assert database.get_inventory("ITEM-LAPTOP-15")["stock_count"] == 3
    assert database.get_order("ORD-1001")["order_status"] == "exchange_confirmed"
    assert {r["refund_reference"] for r in database.list_refunds()} == SEED_REFUND_REFS


def test_ineligible_order_goes_to_refund(agent):
    result = run(agent, "Order ORD-1003 is broken, I want my money back.")
    seq = tool_sequence(result)
    assert "process_refund" in seq
    assert "confirm_exchange" not in seq


@pytest.mark.parametrize("order_id", ["ORD-1001", "ORD-1005"])
def test_every_success_ends_with_notification_and_final_text(agent, order_id):
    result = run(agent, f"Order {order_id} arrived damaged, please resolve it.")
    seq = tool_sequence(result)
    assert seq[-1] == "send_notification"
    assert result.final_answer.strip()
    assert not result.escalated
    assert len(database.list_notifications()) == 1
    assert database.list_notifications()[0]["delivered"] is True


def test_iteration_count_and_default_guardrail(agent):
    """Happy path uses 5 model calls, so the default of 6 is never hit."""
    result = run(agent, "Order ORD-1005 arrived damaged.")
    assert result.iterations == 5
    assert DEFAULT_MAX_ITERATIONS == 6
    assert not result.escalated


# ---------------------------------------------------------------------------
# Error handling through the loop
# ---------------------------------------------------------------------------

def test_invalid_order_no_crash_no_refund_mentions_not_found(agent):
    result = run(agent, "Order ORD-9999 never arrived, resolve it!")
    seq = tool_sequence(result)

    assert "lookup_order" in seq
    assert "process_refund" not in seq
    assert "confirm_exchange" not in seq
    assert not result.escalated  # graceful, not an escalation
    assert "couldn't find" in result.final_answer.lower()
    assert {r["refund_reference"] for r in database.list_refunds()} == SEED_REFUND_REFS
    # The error payload was fed back to the model as an observation.
    lookup_obs = next(
        obs
        for entry in result.trace
        for obs in entry["observation"]
        if isinstance(obs, dict)
        and obs.get("error_code") == "ORDER_NOT_FOUND"
    )
    assert lookup_obs["status"] == "error"


def test_already_refunded_order_is_not_refunded_again(agent):
    result = run(agent, "Order ORD-1004 was damaged, please refund me.")
    refunds_before = {r["refund_reference"] for r in database.list_refunds()}
    assert refunds_before == {"REF-LEGACY42"}

    seq = tool_sequence(result)
    assert "process_refund" in seq
    assert database.list_refunds()[0]["refund_reference"] == "REF-LEGACY42"
    assert len(database.list_refunds()) == 1

    observations = [obs for entry in result.trace for obs in entry["observation"]]
    assert any(
        isinstance(o, dict) and o.get("error_code") == "ALREADY_REFUNDED" for o in observations
    )


# ---------------------------------------------------------------------------
# Registry security and validation
# ---------------------------------------------------------------------------

def test_registry_is_an_exact_whitelist():
    assert set(TOOL_REGISTRY) == {
        "lookup_order",
        "check_inventory",
        "process_refund",
        "confirm_exchange",
        "send_notification",
    }


def test_unauthorized_tool_is_rejected_without_execution():
    output = dispatch("delete_all_orders", {"confirm": "yes"})
    assert output["status"] == "error"
    assert output["error_code"] == "UNAUTHORIZED_TOOL"
    assert "not permitted" in output["message"]
    assert {r["refund_reference"] for r in database.list_refunds()} == SEED_REFUND_REFS


def test_malformed_args_return_invalid_arguments():
    output = dispatch("lookup_order", {"wrong_field": "ORD-1001"})
    assert output["error_code"] == "INVALID_ARGUMENTS"
    assert "order_id" in output["message"]

    output = dispatch("lookup_order", {"order_id": ""})
    assert output["error_code"] == "INVALID_ARGUMENTS"

    output = dispatch("lookup_order", "not-a-json-object")
    assert output["error_code"] == "INVALID_ARGUMENTS"


def test_json_string_args_are_accepted():
    output = dispatch("lookup_order", json.dumps({"order_id": "ORD-1001"}))
    assert output["status"] == "success"
    assert output["customer_email"] == "ana.silva@example.com"


def test_tool_execution_error_is_wrapped_not_raised(monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(TOOL_REGISTRY, "lookup_order", explode)
    output = dispatch("lookup_order", {"order_id": "ORD-1001"})
    assert output["error_code"] == "TOOL_EXECUTION_ERROR"
    assert "boom" in output["message"]


# ---------------------------------------------------------------------------
# Tool contracts
# ---------------------------------------------------------------------------

def test_tools_never_raise_and_always_have_status():
    from tools import (
        check_inventory,
        confirm_exchange,
        lookup_order,
        process_refund,
        send_notification,
    )

    results = [
        lookup_order("ORD-DOES-NOT-EXIST"),
        check_inventory("ITEM-UNKNOWN"),
        process_refund("ORD-DOES-NOT-EXIST", "reason"),
        confirm_exchange("ORD-DOES-NOT-EXIST"),
        send_notification("not-an-email", "hello"),
    ]
    for result in results:
        assert result["status"] == "error"
        assert "error_code" in result and "message" in result


def test_refund_updates_db_and_exchange_decrements_stock():
    from tools import check_inventory, confirm_exchange, process_refund

    refund = process_refund("ORD-1002", "Damaged item, no stock")
    assert refund["status"] == "success"
    assert refund["refund_reference"].startswith("REF-")
    assert refund["refunded_amount"] == 249.00
    assert database.get_order("ORD-1002")["order_status"] == "refunded"

    exchange = confirm_exchange("ORD-1005")
    assert exchange["status"] == "success"
    assert exchange["exchange_reference"].startswith("EXG-")
    assert database.get_inventory("ITEM-KEYBOARD")["stock_count"] == 6

    assert check_inventory("ITEM-KETTLE")["available"] is False


def test_double_refund_blocked():
    from tools import process_refund

    first = process_refund("ORD-1002", "first")
    second = process_refund("ORD-1002", "second")
    assert first["status"] == "success"
    assert second["status"] == "error"
    assert second["error_code"] == "ALREADY_REFUNDED"
    assert len([r for r in database.list_refunds() if r["order_id"] == "ORD-1002"]) == 1


# ---------------------------------------------------------------------------
# Guardrails: max iterations -> escalation
# ---------------------------------------------------------------------------

def test_runaway_loop_escalates(agent_kwargs=None):
    stuck_agent = ResolutionAgent(llm=MockLLM(loop_forever=True), max_iterations=3)
    result = stuck_agent.run("Please resolve order ORD-1001, it arrived damaged.")

    assert result.escalated is True
    assert result.final_answer == ESCALATION_MESSAGE == "Flagged for human escalation"
    assert result.iterations == 3
    # Nothing was mutated beyond lookups.
    assert {r["refund_reference"] for r in database.list_refunds()} == SEED_REFUND_REFS
    assert not database.list_exchanges()
    assert not database.list_notifications()


def test_iteration_cap_is_respected():
    stuck_agent = ResolutionAgent(llm=MockLLM(loop_forever=True), max_iterations=2)
    result = stuck_agent.run("Order ORD-1001 damaged.")
    assert result.iterations == 2
    assert result.escalated


def test_negative_max_iterations_rejected():
    with pytest.raises(ValueError):
        ResolutionAgent(llm=MockLLM(), max_iterations=0)
