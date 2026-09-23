"""Every way to call a LangChain `Runnable` (chat model, chain, or prompt):

- `invoke`    : one input in, one output out, synchronous
- `batch`     : many inputs in parallel, list of outputs out
- `stream`    : synchronous generator of output chunks (token-by-token)
- `ainvoke`   : async version of invoke
- `abatch`    : async version of batch
- `astream`   : async version of stream

All of these work identically whether called on a raw chat model, a prompt
template, or a full `prompt | model | parser` chain, because they all
implement the same `Runnable` interface.
"""

import asyncio

from langchain_core.prompts import ChatPromptTemplate

from core import get_model


def main() -> None:
    model = get_model()
    prompt = ChatPromptTemplate.from_template("In one short sentence, define {term}.")
    chain = prompt | model

    # 1. invoke: single call, blocks until the full response is back.
    print("--- invoke ---")
    result = chain.invoke({"term": "recursion"})
    print(result.content)

    # 2. batch: fan out several inputs concurrently, get a list back in order.
    print("\n--- batch ---")
    terms = ["closure", "polymorphism", "idempotence"]
    results = chain.batch([{"term": t} for t in terms])
    for term, r in zip(terms, results):
        print(f"[{term}] {r.content}")

    # 3. stream: consume the response incrementally as chunks arrive.
    print("\n--- stream ---")
    for chunk in chain.stream({"term": "backpropagation"}):
        print(chunk.content, end="", flush=True)
    print()

    # 4. async variants: ainvoke / abatch / astream, same semantics, non-blocking.
    async def run_async() -> None:
        print("\n--- ainvoke ---")
        result = await chain.ainvoke({"term": "concurrency"})
        print(result.content)

        print("\n--- abatch ---")
        async_terms = ["mutex", "deadlock"]
        results = await chain.abatch([{"term": t} for t in async_terms])
        for term, r in zip(async_terms, results):
            print(f"[{term}] {r.content}")

        print("\n--- astream ---")
        async for chunk in chain.astream({"term": "coroutine"}):
            print(chunk.content, end="", flush=True)
        print()

    asyncio.run(run_async())


if __name__ == "__main__":
    main()
