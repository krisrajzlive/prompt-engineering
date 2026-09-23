"""Side-by-side comparison of core prompt-engineering techniques.

Same underlying task (a word problem), four different prompting styles, run
against the same model so the *prompt* is the only variable:

- zero-shot: instruction only, no examples
- few-shot: a couple of worked examples first
- chain-of-thought: explicitly asked to reason step by step before answering
- role-based: framed through a persona/expert role
"""

from langchain_core.prompts import ChatPromptTemplate

from core import get_model

QUESTION = (
    "A bakery sells cupcakes in boxes of 6. If they baked 134 cupcakes, "
    "how many full boxes can they fill and how many cupcakes are left over?"
)

TECHNIQUES = {
    "zero-shot": ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question directly with just the final result."),
            ("human", "{question}"),
        ]
    ),
    "few-shot": ChatPromptTemplate.from_messages(
        [
            ("system", "Answer in the same format as the examples: 'X full boxes, Y left over.'"),
            ("human", "48 items in boxes of 5."),
            ("ai", "9 full boxes, 3 left over."),
            ("human", "77 items in boxes of 10."),
            ("ai", "7 full boxes, 7 left over."),
            ("human", "{question}"),
        ]
    ),
    "chain-of-thought": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Think through the problem step by step, showing your division and "
                "remainder calculation, then give the final answer on the last line "
                "prefixed with 'Answer:'.",
            ),
            ("human", "{question}"),
        ]
    ),
    "role-based": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a meticulous bakery operations manager who always double-checks "
                "packaging math before reporting it to staff. Be brief but precise.",
            ),
            ("human", "{question}"),
        ]
    ),
}


def main() -> None:
    model = get_model()
    for name, prompt in TECHNIQUES.items():
        rendered = prompt.invoke({"question": QUESTION})
        response = model.invoke(rendered)
        print(f"--- {name} ---")
        print(response.content)
        print()


if __name__ == "__main__":
    main()
