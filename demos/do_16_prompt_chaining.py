"""Prompt chaining: decomposing one task into sequential LLM calls.

This is a different technique from chain-of-thought (see
demos/do_06_prompt_engineering_techniques.py), even though both have
"chain" in the name:

- Chain-of-thought: ONE LLM call, asked to reason step by step *within*
  that single response.
- Prompt chaining: MULTIPLE LLM calls, where call N's output becomes part
  of the input to call N+1's prompt. Each step gets its own focused
  prompt (and could even use a different model), and the intermediate
  results are inspectable/loggable between steps.

Task here: turn a paragraph into a tightened rebuttal, in three chained
steps — extract key claims, draft a rebuttal from those claims, then
compress the draft to two sentences. Shown two ways: manual `.invoke()`
calls (so you can print/inspect each intermediate result), then the same
pipeline as a single piped LCEL chain.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core import get_model

PARAGRAPH = (
    "Remote work reduces company culture and collaboration. Employees who "
    "work from home miss spontaneous hallway conversations that spark new "
    "ideas, and junior staff in particular lose out on the informal "
    "mentorship that happens when everyone shares an office."
)

EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "List the key claims made in this passage as a short bulleted list, "
    "one claim per line, no commentary:\n\n{passage}"
)
DRAFT_PROMPT = ChatPromptTemplate.from_template(
    "Write a rebuttal that addresses each of these claims individually:\n\n{claims}"
)
TIGHTEN_PROMPT = ChatPromptTemplate.from_template(
    "Compress this rebuttal to exactly two sentences, keeping the strongest points:\n\n{draft}"
)


def main() -> None:
    model = get_model()
    parser = StrOutputParser()

    # --- Manual version: each step is its own .invoke() call, so the
    #     intermediate output is visible and could be logged, validated,
    #     or even routed to a different model before the next step. ---
    print("--- Step 1: extract claims ---")
    claims = (EXTRACT_PROMPT | model | parser).invoke({"passage": PARAGRAPH})
    print(claims)

    print("\n--- Step 2: draft rebuttal (from step 1's output) ---")
    draft = (DRAFT_PROMPT | model | parser).invoke({"claims": claims})
    print(draft)

    print("\n--- Step 3: tighten to two sentences (from step 2's output) ---")
    final = (TIGHTEN_PROMPT | model | parser).invoke({"draft": draft})
    print(final)

    # --- Same pipeline as one piped LCEL chain. Each lambda's parameter
    #     (e.g. `parser_output`) is a fresh local name bound to whatever the
    #     PRECEDING pipe stage just produced — it does NOT refer to the
    #     `claims`/`draft` variables above; those only exist in the manual
    #     version. The lambda's only job is to wrap that string into the
    #     dict shape the next prompt's {variable} expects. Less visible
    #     mid-chain than the manual version, but composes into a single
    #     Runnable you can .batch() or .stream() as one unit. ---
    chained = (
        EXTRACT_PROMPT
        | model
        | parser
        | (lambda parser_output: {"claims": parser_output})
        | DRAFT_PROMPT
        | model
        | parser
        | (lambda parser_output: {"draft": parser_output})
        | TIGHTEN_PROMPT
        | model
        | parser
    )
    print("\n--- Same 3 steps as one piped LCEL chain ---")
    print(chained.invoke({"passage": PARAGRAPH}))


if __name__ == "__main__":
    main()
