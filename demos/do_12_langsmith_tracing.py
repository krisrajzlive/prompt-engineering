"""LangSmith tracing: observe prompts, chains and token usage.

Setting `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` (already in .env) is
enough — LangChain auto-instruments every Runnable, no code changes needed.
This demo just runs a small chain and a batch so there's something to look
at, and prints the project name traces are landing in.
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from core import get_model


@traceable(name="classify_sentiment")
def classify_sentiment(text: str) -> str:
    """A custom function wrapped with @traceable shows up as its own span."""
    model = get_model()
    prompt = ChatPromptTemplate.from_template(
        "Classify the sentiment of this text as positive, negative, or neutral. "
        "Reply with one word only.\n\nText: {text}"
    )
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"text": text}).strip()


def main() -> None:
    tracing_enabled = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    print(f"LangSmith tracing enabled: {tracing_enabled} (project: {project})")
    if not tracing_enabled:
        print("Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to see traces.")

    reviews = [
        "This product completely exceeded my expectations!",
        "Terrible experience, would not recommend.",
        "It arrived on time and works as described.",
    ]
    for review in reviews:
        sentiment = classify_sentiment(review)
        print(f"[{sentiment:>8}] {review}")

    if tracing_enabled:
        print(f"\nView the trace at: https://smith.langchain.com (project: {project})")


if __name__ == "__main__":
    main()
