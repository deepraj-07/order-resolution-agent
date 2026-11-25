"""Production-safe request optimization primitives for the resolution agent.

All counters are measurements from this process; unavailable provider values stay
``None`` so callers can render them as N/A rather than inventing improvements.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


def env_flag(name: str, default: bool = True) -> bool:
    return os.getenv(name, str(default)).lower() not in {"0", "false", "no", "off"}


@dataclass
class OptimizationConfig:
    enabled: bool = field(default_factory=lambda: env_flag("AI_OPTIMIZATION_ENABLED"))
    prompt_compaction: bool = field(default_factory=lambda: env_flag("PROMPT_COMPACTION_ENABLED"))
    payload_pruning: bool = field(default_factory=lambda: env_flag("PAYLOAD_PRUNING_ENABLED"))
    parallel_tools: bool = field(default_factory=lambda: env_flag("PARALLEL_TOOL_CALLING"))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "120")))
    cache_max_entries: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_ENTRIES", "500")))
    max_history_messages: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "6")))
    tool_timeout_ms: int = field(default_factory=lambda: int(os.getenv("TOOL_TIMEOUT_MS", "10000")))


@dataclass
class Metrics:
    original_payload_bytes: int = 0
    optimized_payload_bytes: int = 0
    original_input_tokens: Optional[int] = None
    optimized_input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_latency_ms: List[float] = field(default_factory=list)
    uncached_latency_ms: List[float] = field(default_factory=list)
    tool_latency_ms: Dict[str, float] = field(default_factory=dict)
    tools_executed: int = 0
    tool_failures: int = 0
    parallel_latency_ms: Optional[float] = None
    sequential_latency_ms: Optional[float] = None
    started: float = field(default_factory=time.perf_counter)

    def snapshot(self) -> Dict[str, Any]:
        def pct(old: float, new: float) -> Optional[float]:
            return round((old - new) * 100 / old, 2) if old else None
        hits, misses = self.cache_hits, self.cache_misses
        return {
            "original_payload_bytes": self.original_payload_bytes or None,
            "optimized_payload_bytes": self.optimized_payload_bytes or None,
            "payload_reduction_pct": pct(self.original_payload_bytes, self.optimized_payload_bytes),
            "original_input_tokens": self.original_input_tokens,
            "optimized_input_tokens": self.optimized_input_tokens,
            "output_tokens": self.output_tokens,
            "token_savings_pct": pct(self.original_input_tokens or 0, self.optimized_input_tokens or 0),
            "cache_hits": hits, "cache_misses": misses,
            "cache_hit_ratio_pct": round(hits * 100 / (hits + misses), 2) if hits + misses else None,
            "cache_ttl_seconds": CONFIG.cache_ttl_seconds,
            "cached_latency_ms": round(sum(self.cache_latency_ms) / len(self.cache_latency_ms), 2) if self.cache_latency_ms else None,
            "uncached_latency_ms": round(sum(self.uncached_latency_ms) / len(self.uncached_latency_ms), 2) if self.uncached_latency_ms else None,
            "sequential_latency_ms": self.sequential_latency_ms,
            "parallel_latency_ms": self.parallel_latency_ms,
            "tool_speed_improvement_pct": (pct(self.sequential_latency_ms, self.parallel_latency_ms)
                                           if self.sequential_latency_ms is not None and self.parallel_latency_ms is not None else None),
            "tools_executed": self.tools_executed, "tool_failures": self.tool_failures,
            "gemini_api_calls": self.api_calls,
            "end_to_end_ms": round((time.perf_counter() - self.started) * 1000, 2),
        }


class TTLCache:
    """Bounded process-local TTL cache with single-flight miss deduplication.

    It is intentionally not shared across workers. Use Redis in multi-instance
    deployments; this cache never stores failed calls unless the loader chooses to.
    """
    def __init__(self, ttl_seconds: int = 120, max_entries: int = 500) -> None:
        self.ttl, self.max_entries = ttl_seconds, max_entries
        self._values: Dict[str, tuple[float, Any]] = {}
        self._pending: Dict[str, Future] = {}
        self._lock = threading.Lock()
    @staticmethod
    def key(operation: str, args: Dict[str, Any], tenant: str = "") -> str:
        raw = json.dumps([tenant, operation, args], sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
    def get_or_load(self, key: str, loader: Callable[[], Dict[str, Any]], metrics: Optional[Metrics] = None) -> Dict[str, Any]:
        now, owner = time.monotonic(), False
        with self._lock:
            self._values = {k: v for k, v in self._values.items() if v[0] > now}
            hit = self._values.get(key)
            if hit:
                if metrics: metrics.cache_hits += 1
                return hit[1]
            future = self._pending.get(key)
            if future is None:
                future, owner = Future(), True
                self._pending[key] = future
        if not owner:
            if metrics: metrics.cache_hits += 1
            return future.result()
        if metrics: metrics.cache_misses += 1
        try:
            value = loader()
            if value.get("status") == "success":
                with self._lock:
                    if len(self._values) >= self.max_entries: self._values.pop(next(iter(self._values)))
                    self._values[key] = (time.monotonic() + self.ttl, value)
            future.set_result(value)
            return value
        except Exception as exc:
            future.set_exception(exc); raise
        finally:
            with self._lock: self._pending.pop(key, None)
    def delete(self, key: str) -> None:
        with self._lock: self._values.pop(key, None)
    def clear(self) -> None:
        with self._lock: self._values.clear()


def json_size(value: Any) -> int:
    return len(json.dumps(value, default=str, separators=(",", ":")).encode())

def prune(value: Any, allowed: Optional[Iterable[str]] = None, max_text: int = 2000) -> Any:
    """Remove empty noise while preserving 0/False and required tool protocol data."""
    if isinstance(value, dict):
        return {k: prune(v, max_text=max_text) for k, v in value.items()
                if (allowed is None or k in allowed) and v is not None and v != "" and v != {} and v != []}
    if isinstance(value, list): return [prune(v, max_text=max_text) for v in value]
    return value[:max_text] if isinstance(value, str) and len(value) > max_text else value

CONFIG = OptimizationConfig()
CACHE = TTLCache(CONFIG.cache_ttl_seconds, CONFIG.cache_max_entries)
