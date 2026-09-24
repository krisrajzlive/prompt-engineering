"""Response caching (disk): a cache that survives process restarts.

Same `set_llm_cache()` API as demos/do_17_response_caching.py, just a
different backend: `SQLiteCache` writes each (prompt, params) -> output
pair to a `.sqlite` file instead of an in-memory dict. That means a cache
hit works even in a brand new process, as long as it points at the same
database file -- useful for repeated local dev runs against the same
prompts without re-spending tokens every time you restart the script.

Run this script once to create the cache file, then run it again (without
deleting anything) to see a cache hit survive into a brand new process --
that's the part `InMemoryCache` can never do, since it starts empty every
run. `langchain_community`'s internal (de)serializer emits a pending-
deprecation warning unrelated to caching itself; suppressed below since
it's noise for this demo, not something this code can configure away.
"""

import time
import warnings
from pathlib import Path

from langchain_community.cache import SQLiteCache
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
from langchain_core.globals import set_llm_cache

from core import get_model

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

CACHE_PATH = Path(__file__).parent.parent / ".cache" / "responses.sqlite"


def main() -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    is_first_run = not CACHE_PATH.exists()

    set_llm_cache(SQLiteCache(database_path=str(CACHE_PATH)))
    model = get_model()

    prompt = "What is the capital of Germany?"

    start = time.monotonic()
    first = model.invoke(prompt)
    first_elapsed = time.monotonic() - start
    print(f"Call 1 ({first_elapsed:.3f}s): {first.content}")

    # Same prompt again, same process -> cache hit (works identically to
    # the in-memory version at this point).
    start = time.monotonic()
    second = model.invoke(prompt)
    second_elapsed = time.monotonic() - start
    print(f"Call 2 ({second_elapsed:.3f}s, cached): {second.content}\n")

    if is_first_run:
        print(f"Cache written to {CACHE_PATH}")
        print("Run this script again (without deleting the file) to see the "
              "cache hit survive across a brand new process.")
    else:
        print(f"Loaded existing cache from {CACHE_PATH}")
        print("Both calls above were served from a PREVIOUS process's cache "
              "-- that's the difference from InMemoryCache, which starts empty every run.")


if __name__ == "__main__":
    main()
