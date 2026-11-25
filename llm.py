"""LLM backends behind a provider-agnostic interface.

- :class:`GeminiLLM` calls Google Gemini via the new ``google-genai`` SDK with
  manual function calling (automatic function calling disabled) and retries
  transient API failures (429 / 5xx) with exponential backoff.
- :class:`MockLLM` is a deterministic, offline rule-based stand-in that walks
  the same resolution flow: lookup_order -> check_inventory ->
  (process_refund | confirm_exchange) -> send_notification -> final text.

Both return an :class:`LLMResponse` so ``agent`` never touches provider code.
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from schemas import build_gemini_tools
from optimization import Metrics

logger = logging.getLogger(__name__)

try:  # google-genai is a hard dependency in practice; degrade gracefully
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - only hit without the SDK installed
    genai_types = None

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM backend fails after all retries."""


@dataclass
class ToolCall:
    """One requested tool invocation (name + arguments)."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalized model turn.

    ``raw_content`` is the provider's turn object (for Gemini: the candidate
    ``types.Content`` including thought signatures) and must be appended to
    the conversation history unchanged.
    """

    text: str
    function_calls: List[ToolCall]
    raw_content: Any
    thought: Optional[str] = None


class BaseLLM(ABC):
    """Common interface used by the agent loop."""

    @abstractmethod
    def generate(self, contents: List[Any], metrics: Metrics | None = None) -> LLMResponse:
        """Given the ordered conversation ``contents``, produce the next turn."""


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """True for transient HTTP failures (429 rate limit, 5xx, timeouts)."""
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS:
        return True
    message = str(exc)
    return any(token in message for token in ("429", "503", "Resource", "unavailable", "UNAVAILABLE"))


class GeminiLLM(BaseLLM):
    """Gemini backend using manual (agent-driven) function calling."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        retry_delays: tuple = (1.0, 3.0),
    ) -> None:
        from google import genai  # imported lazily: not needed for MockLLM runs

        key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file or pass api_key=..."
            )
        # Function calling requires the v1beta API version for some key types
        # (otherwise: 400 "Function calling is not enabled for api version v1").
        self._client = genai.Client(
            api_key=key,
            http_options=genai.types.HttpOptions(api_version="v1beta"),
        )
        self._model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self._tools = build_gemini_tools()
        self._retry_delays = retry_delays

    def generate(self, contents: List[Any], metrics: Metrics | None = None) -> LLMResponse:
        """Call Gemini, retrying up to 2 times on transient failures."""
        config = self._build_config()
        last_exc: Exception | None = None
        for attempt in range(len(self._retry_delays) + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=list(contents),
                    config=config,
                )
                if metrics:
                    metrics.api_calls += 1
                    usage = getattr(response, "usage_metadata", None)
                    if usage:
                        metrics.optimized_input_tokens = (metrics.optimized_input_tokens or 0) + (getattr(usage, "prompt_token_count", 0) or 0)
                        metrics.output_tokens = (metrics.output_tokens or 0) + (getattr(usage, "candidates_token_count", 0) or 0)
                return self._parse_response(response)
            except Exception as exc:  # noqa: BLE001 - classified below
                if _is_retryable(exc) and attempt < len(self._retry_delays):
                    delay = self._retry_delays[attempt]
                    logger.warning(
                        "Gemini call failed (%s); retrying in %.1fs (attempt %d)",
                        exc,
                        delay,
                        attempt + 1,
                    )
                    time.sleep(delay)
                    last_exc = exc
                    continue
                raise LLMUnavailableError(str(exc)) from exc
        raise LLMUnavailableError(str(last_exc))  # pragma: no cover

    def _build_config(self) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=agent_system_prompt(),
            tools=self._tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0,
        )

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        calls: List[ToolCall] = []
        for fc in response.function_calls or []:
            args = dict(fc.args) if fc.args else {}
            calls.append(ToolCall(name=fc.name, args=args))

        raw = None
        try:
            if response.candidates:
                raw = response.candidates[0].content
        except (IndexError, AttributeError):
            raw = None
        if raw is None and calls:
            # Synthetic fallback: unusual with real Gemini responses,
            # but keeps the loop correct if the SDK omits the candidate.
            if genai_types is not None:
                raw = genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(function_call=genai_types.FunctionCall(name=c.name, args=c.args))
                        for c in calls
                    ],
                )
            else:
                raw = {
                    "role": "model",
                    "parts": [{"function_call": {"name": c.name, "args": c.args}} for c in calls],
                }

        text = ""
        try:
            text = response.text or ""
        except Exception:  # noqa: BLE001 - .text can raise on blocked prompts
            text = ""
        return LLMResponse(text=text, function_calls=calls, raw_content=raw)


def agent_system_prompt() -> str:
    """System prompt (kept in llm.py so both backends share the wording)."""
    return (
        "Resolve damaged-order requests with tools. Lookup order, then inventory. "
        "Exchange only if eligible and in stock; otherwise refund with a brief reason. "
        "Notify after a successful resolution, then give a concise summary. "
        "For missing orders do not mutate: ask for a valid ID. Use only tool facts; adapt to errors."
    )


# ---------------------------------------------------------------------------
# Mock backend (offline, deterministic, same tool-call discipline)
# ---------------------------------------------------------------------------

_ORDER_ID_RE = re.compile(r"ORD-\d+", re.IGNORECASE)


def _mock_error(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "error_code": code, "message": message}


class MockLLM(BaseLLM):
    """Rule-based LLM stand-in that follows the required tool sequence.

    Each call inspects the conversation history (which tool responses already
    exist) and emits the next step, mirroring Gemini's manual function-calling
    behavior. Requires no network.
    """

    def __init__(self, loop_forever: bool = False) -> None:
        self.loop_forever = loop_forever

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _call(name: str, args: Dict[str, Any], thought: str) -> LLMResponse:
        raw: Any
        if genai_types is not None:
            raw = genai_types.Content(
                role="model",
                parts=[genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args))],
            )
        else:  # pragma: no cover - only without the SDK installed
            raw = {"role": "model", "parts": [{"function_call": {"name": name, "args": args}}]}
        return LLMResponse(text="", function_calls=[ToolCall(name=name, args=args)], raw_content=raw, thought=thought)

    @staticmethod
    def _final(text: str, thought: str) -> LLMResponse:
        raw: Any
        if genai_types is not None:
            raw = genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
        else:  # pragma: no cover
            raw = {"role": "model", "parts": [{"text": text}]}
        return LLMResponse(text=text, function_calls=[], raw_content=raw, thought=thought)

    @staticmethod
    def _extract(contents: List[Any]) -> tuple[str, Dict[str, Dict[str, Any]]]:
        """Return (first user message, latest tool responses by name)."""
        user_message = ""
        latest: Dict[str, Dict[str, Any]] = {}
        for content in contents:
            if genai_types is not None and isinstance(content, genai_types.Content):
                role = content.role
                parts = content.parts or []
                parts = [
                    {
                        "function_response": (
                            {"name": p.function_response.name, "response": p.function_response.response}
                            if p.function_response
                            else None
                        ),
                        "text": p.text,
                    }
                    for p in parts
                ]
            elif isinstance(content, dict):
                role = content.get("role")
                parts = content.get("parts", [])
            else:
                continue
            for part in parts:
                fr = part.get("function_response")
                if fr:
                    response_payload = fr.get("response") or {}
                    latest[fr.get("name", "")] = response_payload.get("result", response_payload)
                text = part.get("text")
                if role == "user" and text and not user_message:
                    user_message = text
        return user_message, latest

    # -- interface ----------------------------------------------------------

    def generate(self, contents: List[Any], metrics: Metrics | None = None) -> LLMResponse:
        user_message, latest = self._extract(contents)
        done = set(latest)

        if self.loop_forever:  # guardrail demo: never finishes
            return self._call(
                "lookup_order",
                {"order_id": "ORD-1001"},
                "I keep re-checking the order no matter what happens.",
            )

        match = _ORDER_ID_RE.search(user_message or "")
        order_id = match.group(0).upper() if match else ""

        if not order_id:
            return self._final(
                "Could you share your order ID (it looks like ORD-1234)? "
                "Then I can investigate the issue right away.",
                "no order ID in the message; ask for it.",
            )

        if "lookup_order" not in done:
            return self._call(
                "lookup_order",
                {"order_id": order_id},
                f"I need the details of {order_id} before acting.",
            )

        lookup = latest["lookup_order"]
        if lookup.get("status") == "error":
            return self._final(
                f"I couldn't find order {order_id} in our system. Please verify "
                "the order ID and I'll take another look.",
                "the order does not exist; do not refund, ask for verification.",
            )

        if "check_inventory" not in done:
            return self._call(
                "check_inventory",
                {"item_id": lookup["item_id"]},
                f"checking stock for {lookup['item_id']} to decide "
                "between exchange and refund.",
            )

        resolution_done = done & {"process_refund", "confirm_exchange"}
        if not resolution_done:
            inventory = latest["check_inventory"]
            eligible = bool(lookup.get("exchange_eligible")) and bool(inventory.get("available"))
            if eligible:
                return self._call(
                    "confirm_exchange",
                    {"order_id": order_id},
                    "item is eligible and in stock -> confirm the exchange.",
                )
            reason = self._refund_reason(user_message, lookup, inventory)
            return self._call(
                "process_refund",
                {"order_id": order_id, "reason": reason},
                "exchange impossible -> refund to the original payment method.",
            )

        if "send_notification" not in done:
            resolution_name = "confirm_exchange" if "confirm_exchange" in done else "process_refund"
            resolution = latest[resolution_name]
            email = lookup["customer_email"]
            name = lookup["customer_name"]
            message = self._notification_message(order_id, name, resolution_name, resolution)
            return self._call(
                "send_notification",
                {"customer_email": email, "message": message},
                "resolution is done; email a confirmation.",
            )

        # All tools done -> final summary.
        return self._final(self._summary(order_id, lookup, latest), "everything is resolved; summarize.")

    # -- message builders ----------------------------------------------------

    @staticmethod
    def _refund_reason(user_message: str, lookup: Dict[str, Any], inventory: Dict[str, Any]) -> str:
        lowered = (user_message or "").lower()
        if not inventory.get("available"):
            base = "Replacement unit is out of stock"
        elif not lookup.get("exchange_eligible"):
            base = f"Order not exchange eligible ({lookup.get('eligibility_reason', 'policy')})"
        else:
            base = "Customer requested resolution"
        context = "damaged item" if "damag" in lowered else ("defective item" if "defect" in lowered else "customer issue")
        return f"{base}: {context}."

    @staticmethod
    def _notification_message(
        order_id: str, name: str, kind: str, resolution: Dict[str, Any]
    ) -> str:
        if kind == "confirm_exchange" and resolution.get("status") == "success":
            return (
                f"Dear {name}, your replacement for order {order_id} is confirmed "
                f"(exchange reference {resolution['exchange_reference']}). "
                "We will ship it right away."
            )
        if kind == "process_refund" and resolution.get("status") == "success":
            return (
                f"Dear {name}, we refunded {resolution['refunded_amount']:.2f} USD "
                f"for order {order_id} to {resolution['refunded_to']} "
                f"(refund reference {resolution['refund_reference']})."
            )
        return (
            f"Dear {name}, we reviewed order {order_id}: "
            f"{resolution.get('message', 'your case needs manual follow-up')}. "
            "A specialist will contact you shortly."
        )

    @staticmethod
    def _summary(order_id: str, lookup: Dict[str, Any], latest: Dict[str, Any]) -> str:
        resolution_name = "confirm_exchange" if "confirm_exchange" in latest else "process_refund"
        resolution = latest[resolution_name]
        notified = latest.get("send_notification", {})
        email = lookup.get("customer_email", "")
        if resolution_name == "confirm_exchange" and resolution.get("status") == "success":
            body = (
                f"Your exchange for order {order_id} is confirmed under reference "
                f"{resolution['exchange_reference']} (remaining stock: "
                f"{resolution['remaining_stock']})."
            )
        elif resolution.get("status") == "success":
            body = (
                f"A refund of {resolution['refunded_amount']:.2f} USD for order "
                f"{order_id} was issued to {resolution['refunded_to']} under "
                f"reference {resolution['refund_reference']}."
            )
        elif resolution.get("error_code") == "ALREADY_REFUNDED":
            body = f"Order {order_id} had already been refunded; no new refund was issued."
        else:
            body = (
                f"Order {order_id} needs manual follow-up: "
                f"{resolution.get('message', 'unexpected tool error')}."
            )
        sent = (
            f" A confirmation email has been sent to {email}."
            if notified.get("delivered")
            else ""
        )
        greeting = f"Hi {lookup.get('customer_name', 'there')}! "
        return f"{greeting}{body}{sent} Anything else I can help with?"
