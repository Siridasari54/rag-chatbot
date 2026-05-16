# utils/cache.py  — FULL REPLACEMENT

import hashlib
import time
import streamlit as st

_MAX_CACHE_ENTRIES = 100  # ★ evict oldest when exceeded


def make_query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def get_cached(query: str):
    h     = make_query_hash(query)
    entry = st.session_state.query_cache.get(h)
    if entry and (time.time() - entry["ts"]) < st.session_state.cache_ttl:
        # ★ bump access time so LRU eviction keeps hot entries
        entry["ts_access"] = time.time()
        return entry
    if entry:
        # expired — remove it
        del st.session_state.query_cache[h]
    return None


def set_cache(query: str, answer: str, sources) -> None:
    cache = st.session_state.query_cache

    # ★ Evict oldest entry by access time when over budget
    if len(cache) >= _MAX_CACHE_ENTRIES:
        oldest_key = min(
            cache,
            key=lambda k: cache[k].get("ts_access", cache[k]["ts"])
        )
        del cache[oldest_key]

    h = make_query_hash(query)
    cache[h] = {
        "answer":    answer,
        "sources":   sources,
        "ts":        time.time(),
        "ts_access": time.time(),
    }