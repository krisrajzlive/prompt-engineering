"""Partial prompts and prompt composition.

`partial()` pre-fills some variables (a fixed value, or a zero-arg function
evaluated at render time, e.g. today's date) so callers only supply what
changes per call. `PipelinePromptTemplate`-style composition lets you build
a big prompt out of small, independently testable pieces (a persona block +
a formatting-rules block + the actual task) instead of one giant string.
"""

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from core import get_model


def main() -> None:
    # 1. Partial with a static value.
    base = PromptTemplate.from_template("You work for {company}. Task: {task}")
    partial_static = base.partial(company="Acme Robotics")
    print("--- Partial (static value) ---")
    print(partial_static.format(task="draft a release note"))

    # 2. Partial with a callable, resolved fresh on every .format() call.
    dated = PromptTemplate.from_template("Today is {today}. Task: {task}")
    partial_dynamic = dated.partial(today=lambda: datetime.now().strftime("%Y-%m-%d"))
    print("\n--- Partial (callable, resolved at render time) ---")
    print(partial_dynamic.format(task="summarize today's standup"))

    # 3. Composing a chat prompt from reusable string fragments.
    persona = "You are a senior code reviewer. Be direct and specific."
    rules = "Rules: point out at most 3 issues, cite line numbers, no praise."
    composed = ChatPromptTemplate.from_messages(
        [
            ("system", f"{persona}\n{rules}"),
            ("human", "Review this snippet:\n{code}"),
        ]
    )
    rendered = composed.invoke({"code": "def add(a,b):\n  return a+b"})
    print("\n--- Composed system prompt from persona + rules fragments ---")
    for m in rendered.to_messages():
        print(f"[{m.type}] {m.content}")

    response = get_model().invoke(rendered)
    print("\nModel:", response.content)


if __name__ == "__main__":
    main()
