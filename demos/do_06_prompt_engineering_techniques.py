"""Side-by-side comparison of core prompt-engineering techniques.

Same underlying task (a word problem), five different prompting styles, run
against the same model so the *prompt* is the only variable:

- zero-shot: instruction only, no examples
- few-shot: a couple of worked examples first (answer-only, no reasoning)
- zero-shot chain-of-thought: no examples, just told to reason step by step
  (Kojima et al., 2022 — the "let's think step by step" trick)
- few-shot chain-of-thought: shown a worked example *with its reasoning
  trace*, not just its final answer, so the model imitates the reasoning
  pattern rather than only the output format (Wei et al., 2022 — this is
  what "chain-of-thought prompting" originally meant, and it's a
  meaningfully different lever than zero-shot CoT above)
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
    "zero-shot-cot": ChatPromptTemplate.from_messages(
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
    "few-shot-cot": ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Solve each problem by reasoning step by step like the example, "
                "then give the final answer on the last line prefixed with 'Answer:'.",
            ),
            ("human", "A florist sells roses in bunches of 8. If they picked 101 roses, "
                       "how many full bunches can they make and how many roses are left over?"),
            (
                "ai",
                "To find the number of full bunches, divide the total roses by the bunch size:\n"
                "101 / 8 = 12 remainder 5\n"
                "So 12 bunches use 12 * 8 = 96 roses.\n"
                "Roses left over: 101 - 96 = 5\n"
                "Answer: 12 full bunches, 5 left over.",
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
