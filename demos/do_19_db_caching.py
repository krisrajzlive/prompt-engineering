"""Response caching (db): a cache shared across processes and machines.

Same `set_llm_cache()` API as demos/do_17_response_caching.py and
demos/do_18_disk_caching.py, again just a different backend: `RedisCache`
stores each (prompt, params) -> output pair in a real Redis server instead
of RAM or a local file. That means the cache is shared -- any process that
can reach the same Redis instance gets a hit, not just the process that
wrote it. This is the shape you'd actually use in production: multiple
API server instances behind a load balancer all sharing one cache, an
optional per-entry TTL so stale answers eventually expire, and the cache
surviving any single process restart or deploy.

Requires a Redis server reachable at localhost:6380 (this demo spins up a
disposable one via `docker run -d -p 6380:6379 redis:7`, kept separate
from port 6379 to avoid clashing with any other Redis already running
locally). If nothing is listening there, this fails fast with a clear
connection error rather than silently falling back to no caching.

LangSmith tracing is disabled for this script specifically: its
background flush thread can throw a harmless but noisy
"can't create new thread at interpreter shutdown" RuntimeError right as
a short-lived script like this one exits. Must happen before `core` is
imported, since `core.get_model()` reads LANGSMITH_TRACING from the
environment via python-dotenv (which never overrides a var already set).
"""

import os
import time
import warnings

os.environ["LANGSMITH_TRACING"] = "false"

import redis
from langchain_community.cache import RedisCache
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_core.globals import set_llm_cache

from core import get_model

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

REDIS_HOST = "localhost"
REDIS_PORT = 6380


def main() -> None:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    redis_client.ping()  # fail fast with a clear error if Redis isn't reachable

    # ttl=60: entries auto-expire after 60s, so a real deployment doesn't
    # serve indefinitely stale answers. Omit ttl for cache entries that
    # never expire on their own.
    set_llm_cache(RedisCache(redis_client, ttl=60))
    model = get_model()

    prompt = "What is the capital of Italy?"

    start = time.monotonic()
    first = model.invoke(prompt)
    first_elapsed = time.monotonic() - start
    print(f"Call 1 ({first_elapsed:.3f}s): {first.content}")

    # Same prompt, same process -> cache hit, identical to the in-memory
    # and disk-backed versions at this point.
    start = time.monotonic()
    second = model.invoke(prompt)
    second_elapsed = time.monotonic() - start
    print(f"Call 2 ({second_elapsed:.3f}s, cached): {second.content}\n")

    # What makes this backend different: a SEPARATE Redis client, simulating
    # a different process/server instance, sees the same cache entry --
    # neither InMemoryCache nor a not-yet-shared SQLite file could do this
    # across two live processes without pointing at the same disk.
    other_process_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    set_llm_cache(RedisCache(other_process_client, ttl=60))
    start = time.monotonic()
    from_other_process = model.invoke(prompt)
    other_elapsed = time.monotonic() - start
    print(f"A second Redis client ({other_elapsed:.3f}s, cached): {from_other_process.content}")
    print("-> that second client never made this call before -- it hit the SAME shared cache entry.")


if __name__ == "__main__":
    main()
