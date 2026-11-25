# Optimization report

## Original bottlenecks

The original agent sent verbose tool descriptions and system instructions on
every Gemini turn, executed every function call serially, had no request-size
measurement, and repeated read lookups during a process lifetime.

## Delivered implementation

- `optimization.py`: configuration, schema-aware null/noise pruning, metrics,
  deterministic bounded TTL cache, expiration, and single-flight deduplication.
- `agent.py`: payload measurement, metric handoff, and safe ordered parallel
  execution for a batch containing only read calls.
- `registry.py`: cache integration for `lookup_order` and `check_inventory`.
  Writes bypass it; database reset clears stale cached reads.
- `llm.py` / `schemas.py`: compact prompt and declarations; Gemini usage
  metadata is recorded when supplied.
- `benchmark.py`: controlled-delay concurrency benchmark.

## Measured results

Run `python benchmark.py` for current machine results. It reports actual cache
hits/misses and controlled simulated sequential/parallel timing; it does not
claim those simulated latencies as Gemini performance. Real Gemini input and
output tokens are recorded from usage metadata when available; original-token
comparison is `N/A` until a baseline request is captured.

## Limitations / production recommendation

This CLI project has no web UI, so metrics are exposed on `AgentResult.metrics`
and benchmark JSON instead of a browser panel. The cache is process-local and
must be replaced by Redis for horizontally scaled services. Function calls that
mutate orders remain sequential by design.
