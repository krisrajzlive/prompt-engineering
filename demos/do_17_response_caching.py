"""Response caching (in-memory): LangChain memoizes (prompt, params) -> output.

Note the distinction: this is LangChain's *response* cache — an exact
match on prompt + model params is served from memory instead of calling
the provider again. It is NOT the same thing as provider-side "prompt
caching" (Anthropic/OpenAI caching a long prompt *prefix* server-side to
cut cost while still running a fresh generation) — LangChain's cache
either returns a full cached answer or makes a full fresh call, nothing
in between.

`set_llm_cache(...)` installs a cache globally, for every model instance
in the process, not per-model. `InMemoryCache` (used below) is the
simplest backend -- a dict in RAM, cleared on process exit. See
demos/do_18_disk_caching.py for a persistent, file-backed cache, and
demos/do_19_db_caching.py for a shared, database-backed cache -- same
`set_llm_cache()` call each time, just a different backend object.
"""

import time

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from core import get_model


def main() -> None:
    set_llm_cache(InMemoryCache())
    model = get_model()

    prompt = "What is the capital of France?"

    start = time.monotonic()
    first = model.invoke(prompt)
    first_elapsed = time.monotonic() - start
    print(f"First call  ({first_elapsed:.3f}s): {first.content}")

    # Identical prompt + params -> cache hit, no provider call made.
    # The timing gap here (seconds vs. milliseconds) is the evidence.
    start = time.monotonic()
    second = model.invoke(prompt)
    second_elapsed = time.monotonic() - start
    print(f"Second call ({second_elapsed:.3f}s, cached): {second.content}\n")

    # A different prompt is a cache miss -- it still calls the provider.
    start = time.monotonic()
    third = model.invoke("What is the capital of Japan?")
    third_elapsed = time.monotonic() - start
    print(f"Different prompt ({third_elapsed:.3f}s, NOT cached): {third.content}")


if __name__ == "__main__":
    main()
