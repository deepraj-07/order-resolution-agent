"""Offline, reproducible measurements for cache and parallel tool execution."""
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor
import database
from optimization import CACHE, Metrics
from registry import dispatch

def delayed(name, args):
    time.sleep(.04)  # controlled simulated external latency
    return dispatch(name, args, Metrics())

def main():
    database.reset_db(); CACHE.clear()
    m = Metrics(); dispatch("lookup_order", {"order_id":"ORD-1001"}, m)
    dispatch("lookup_order", {"order_id":"ORD-1001"}, m)
    calls = [("lookup_order", {"order_id":"ORD-1001"}), ("check_inventory", {"item_id":"ITEM-LAPTOP-15"})]
    start=time.perf_counter(); [delayed(*call) for call in calls]; seq=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(lambda c: delayed(*c), calls))
    par=(time.perf_counter()-start)*1000
    print(json.dumps({"simulated_sequential_ms":round(seq,2), "simulated_parallel_ms":round(par,2), "speedup_pct":round((seq-par)*100/seq,2), "cache":m.snapshot()}, indent=2))
if __name__ == "__main__": main()
