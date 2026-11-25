"""The ReAct-style agentic resolution loop with guardrails.

``ResolutionAgent.run`` implements the Thought / Action / Observation cycle:

1. The model turn (Gemini candidate content, thought signatures intact) is
   appended to the history *unchanged*.
2. EVERY requested function call is executed through ``registry.dispatch``.
3. ONE user-role content with one function-response part per call, in the
   same order, is appended (Gemini has no call IDs, so calls are linked by
   name + order).
4. A model turn with no function calls is the final answer.

Guardrails: iteration cap with escalation, tool-error feedback, malformed-arg
self-correction via the dispatcher, and retry/escalation around LLM failures.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List

from google.genai import types

from llm import BaseLLM, LLMUnavailableError
from registry import dispatch
from optimization import CONFIG, Metrics, json_size, prune

logger = logging.getLogger(__name__)

#: Happy path needs 5 model calls (lookup, inventory, resolution, notify,
#: final), so the default cap leaves one spare turn.
DEFAULT_MAX_ITERATIONS = 6

ESCALATION_MESSAGE = "Flagged for human escalation"


@dataclass
class AgentResult:
    """Outcome of one agent run."""

    final_answer: str
    escalated: bool
    iterations: int  # number of model calls actually made
    trace: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def tools_called(self) -> List[str]:
        """Ordered list of tool names executed during the run."""
        return [call["name"] for entry in self.trace for call in entry["action"]]


class ResolutionAgent:
    """Provider-agnostic customer-resolution agent loop."""

    def __init__(
        self,
        llm: BaseLLM,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        verbose: bool = False,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> AgentResult:
        """Run the full ReAct loop for one customer message."""
        metrics = Metrics()
        contents: List[Any] = [
            types.Content(role="user", parts=[types.Part(text=user_message)])
        ]
        trace: List[Dict[str, Any]] = []

        for iteration in range(1, self.max_iterations + 1):
            try:
                payload = {"contents": [str(c) for c in contents]}
                metrics.original_payload_bytes += json_size(payload)
                metrics.optimized_payload_bytes += json_size(prune(payload))
                response = self.llm.generate(contents, metrics=metrics)
            except LLMUnavailableError as exc:
                logger.exception("LLM backend unavailable")
                trace.append(
                    {
                        "iteration": iteration,
                        "thought": f"LLM backend failed: {exc}",
                        "action": [],
                        "observation": [],
                    }
                )
                return self._escalate(trace, iteration, reason=f"llm_error: {exc}", metrics=metrics)

            if response.function_calls:
                contents.append(self._as_content(response.raw_content, response.function_calls))
                response_parts: List[types.Part] = []
                actions: List[Dict[str, Any]] = []
                observations: List[Dict[str, Any]] = []
                calls = response.function_calls
                started = time.perf_counter()
                can_parallelize = CONFIG.enabled and CONFIG.parallel_tools and len(calls) > 1 and all(c.name in {"lookup_order", "check_inventory"} for c in calls)
                if can_parallelize:
                    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                        outputs = list(pool.map(lambda c: dispatch(c.name, c.args, metrics), calls))
                    metrics.parallel_latency_ms = round((time.perf_counter() - started) * 1000, 2)
                else:
                    outputs = [dispatch(c.name, c.args, metrics) for c in calls]
                    metrics.sequential_latency_ms = round((time.perf_counter() - started) * 1000, 2)
                for call, output in zip(calls, outputs):
                    actions.append({"name": call.name, "args": call.args})
                    observations.append(output)
                    metrics.tools_executed += 1
                    metrics.tool_failures += output.get("status") == "error"
                    response_parts.append(
                        self._function_response_part(call.name, output)
                    )
                contents.append(types.Content(role="user", parts=response_parts))
                entry = {
                    "iteration": iteration,
                    "thought": response.thought or "Model requested tool execution.",
                    "action": actions,
                    "observation": observations,
                }
                trace.append(entry)
                self._print_entry(entry)
                continue

            # No function calls -> final answer.
            final_answer = (response.text or "").strip()
            trace.append(
                {
                    "iteration": iteration,
                    "thought": response.thought or "Model produced the final answer.",
                    "action": [],
                    "observation": [final_answer],
                }
            )
            self._print_entry(trace[-1], final=True)
            return AgentResult(
                final_answer=final_answer or "(empty final answer)",
                escalated=False,
                iterations=iteration,
                trace=trace,
                metrics=metrics.snapshot(),
            )

        return self._escalate(trace, self.max_iterations, reason="max_iterations_exceeded", metrics=metrics)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _escalate(self, trace: List[Dict[str, Any]], iterations: int, reason: str, metrics: Metrics | None = None) -> AgentResult:
        """Halt the loop and flag the session for a human."""
        logger.warning(
            "session_flag=escalation reason=%s model_calls=%d", reason, iterations
        )
        trace.append(
            {
                "iteration": iterations,
                "thought": f"Guardrail hit ({reason}); escalating to a human.",
                "action": [],
                "observation": [ESCALATION_MESSAGE],
                "escalated": True,
            }
        )
        self._print_entry(trace[-1], final=True, escalated=True)
        return AgentResult(
            final_answer=ESCALATION_MESSAGE,
            escalated=True,
            iterations=iterations,
            trace=trace,
            metrics=metrics.snapshot() if metrics else {},
        )

    @staticmethod
    def _as_content(raw_content: Any, calls: List[Any]) -> Any:
        """Append the model's turn unchanged, building a fallback if absent."""
        if raw_content is not None:
            if isinstance(raw_content, types.Content):
                return raw_content
            return types.Content.model_validate(raw_content)
        return types.Content(
            role="model",
            parts=[
                types.Part(function_call=types.FunctionCall(name=call.name, args=call.args))
                for call in calls
            ],
        )

    @staticmethod
    def _function_response_part(name: str, output: Dict[str, Any]) -> types.Part:
        """One function-response part, keyed by tool name (Gemini has no IDs)."""
        return types.Part(
            function_response=types.FunctionResponse(name=name, response={"result": output})
        )

    def _print_entry(self, entry: Dict[str, Any], final: bool = False, escalated: bool = False) -> None:
        """Readable Thought / Action / Observation trace."""
        if not self.verbose:
            return
        print(f"\n--- Iteration {entry['iteration']} " + "-" * 50)
        thought = entry.get("thought")
        if thought:
            print(f"[Thought]   {thought}")
        if escalated:
            print(f"[Escalation] {ESCALATION_MESSAGE}")
            return
        if final:
            print(f"[Response]  {entry['observation'][0] if entry['observation'] else ''}")
            return
        for action, observation in zip(entry["action"], entry["observation"]):
            args_preview = ", ".join(f"{k}={v!r}" for k, v in action["args"].items())
            print(f"[Action]      {action['name']}({args_preview})")
            print(f"[Observation] {_short(observation)}")


def _short(payload: Any, limit: int = 240) -> str:
    text = repr(payload)
    return text if len(text) <= limit else text[: limit - 3] + "..."
