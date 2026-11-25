"""CLI entry point for the order resolution agent.

Usage:
    python main.py --demo                 # run the 5 demo scenarios
    python main.py --demo --mock          # identical, forced offline mode
    python main.py --max-iterations 4     # custom guardrail
    python main.py                        # interactive prompt

The Gemini backend is selected whenever GEMINI_API_KEY is present and --mock
is not given; otherwise the offline rule-based MockLLM is used.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import json

from dotenv import load_dotenv

from agent import DEFAULT_MAX_ITERATIONS, ResolutionAgent
import database
from llm import BaseLLM, GeminiLLM, MockLLM


def build_llm(force_mock: bool, max_iterations: int) -> BaseLLM:
    """Pick the backend: Gemini when a key exists, otherwise MockLLM."""
    if not force_mock:
        if os.environ.get("GEMINI_API_KEY", "").strip():
            try:
                llm = GeminiLLM()
                logging.info("Using Gemini backend (%s)", os.getenv("GEMINI_MODEL"))
                return llm
            except Exception as exc:  # noqa: BLE001 - fall back to the mock
                logging.warning("Gemini backend unavailable (%s); using MockLLM.", exc)
        else:
            logging.info("GEMINI_API_KEY not set; using offline MockLLM.")
    return MockLLM()


SCENARIOS = [
    (
        "Damaged item, replacement in stock -> exchange",
        "Hi, my order ORD-1001 arrived with a cracked screen on the laptop. It's damaged, what can you do?",
    ),
    (
        "Damaged item, stock is zero -> refund",
        "The headphones from order ORD-1002 arrived damaged. Can you help?",
    ),
    (
        "Order not exchange-eligible (final sale) -> refund",
        "Order ORD-1003 arrived broken. I'd like my money back please.",
    ),
    (
        "Unknown order -> graceful not-found, no refund",
        "I need help with order ORD-9999, it never arrived.",
    ),
    (
        "Runaway model loop -> iteration cap -> human escalation",
        "Please resolve order ORD-1001, it arrived damaged.",
    ),
]


def _side_effects() -> str:
    refunds = [r["refund_reference"] for r in database.list_refunds()]
    exchanges = [e["exchange_reference"] for e in database.list_exchanges()]
    notifications = [
        f"{n['notification_id']} -> {n['customer_email']}" for n in database.list_notifications()
    ]
    lines = ["DB side effects:"]
    labels = [
        ("refund(s)", refunds, "none"),
        ("exchange(s)", exchanges, "none"),
        ("notification(s)", notifications, "none"),
    ]
    for label, values, empty in labels:
        shown = ", ".join(values) if values else empty
        lines.append(f"  {label}: {shown}")
    return "\n".join(lines)


def run_demo(llm: BaseLLM, max_iterations: int) -> None:
    """Run all demo scenarios, resetting the DB between each."""
    print("=" * 72)
    print("ORDER RESOLUTION AGENT - DEMO")
    print("=" * 72)
    for index, (title, message) in enumerate(SCENARIOS, start=1):
        database.reset_db()
        print(f"\n{'#' * 72}\n# Scenario {index}/{len(SCENARIOS)}: {title}\n{'#' * 72}")
        print(f"\nUser: {message}\n")

        if index == len(SCENARIOS):
            # Guardrail demo: a model that never stops calling tools,
            # regardless of which backend serves the other scenarios.
            loop_llm: BaseLLM = MockLLM(loop_forever=True)
            agent: ResolutionAgent = ResolutionAgent(
                loop_llm, max_iterations=min(max_iterations, 3), verbose=True
            )
        else:
            agent = ResolutionAgent(llm, max_iterations=max_iterations, verbose=True)

        result = agent.run(message)

        print("\n" + "-" * 72)
        print(f"Final answer : {result.final_answer}")
        print(f"Escalated    : {result.escalated}")
        print(f"Model calls  : {result.iterations}")
        print(f"Tools called : {' -> '.join(result.tools_called) or '(none)'}")
        print("Optimization metrics:", json.dumps(result.metrics, default=str))
        print(_side_effects())


def run_interactive(llm: BaseLLM, max_iterations: int) -> None:
    """Interactive prompt; Ctrl+C or 'exit' to leave."""
    database.reset_db()
    print("Order Resolution Agent - interactive mode. Type 'exit' to quit.")
    while True:
        try:
            message = input("\nresolution> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not message:
            continue
        agent = ResolutionAgent(llm, max_iterations=max_iterations, verbose=True)
        result = agent.run(message)
        print(f"\n{result.final_answer}")
        print(f"(tools: {', '.join(result.tools_called) or 'none'})")
        print("(optimization:", json.dumps(result.metrics, default=str), ")")
        print(_side_effects())


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous order resolution agent")
    parser.add_argument("--demo", action="store_true", help="run the demo scenarios")
    parser.add_argument("--mock", action="store_true", help="force the offline MockLLM")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="maximum number of model calls before escalation (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")
    return args


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    args = parse_args(argv)
    llm = build_llm(force_mock=args.mock, max_iterations=args.max_iterations)
    if args.demo:
        run_demo(llm, args.max_iterations)
    else:
        run_interactive(llm, args.max_iterations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
