"""Example selectors: pick few-shot examples dynamically instead of hard-coding them.

With a large example bank, stuffing all of them into every prompt wastes
tokens and can even hurt accuracy. A `SemanticSimilarityExampleSelector`
embeds the examples once, then at call time retrieves only the `k` most
relevant ones for the *current* input — few-shot prompting that scales.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from core import get_embeddings, get_model

EXAMPLES = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is 10 * 3?", "answer": "30"},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
]


def main() -> None:
    example_prompt = PromptTemplate.from_template("Q: {question}\nA: {answer}")

    selector = SemanticSimilarityExampleSelector.from_examples(
        EXAMPLES,
        get_embeddings(),
        FAISS,
        k=2,
    )

    few_shot = FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        prefix="Answer the question in the same terse style as the examples.",
        suffix="Q: {input}\nA:",
        input_variables=["input"],
    )

    for query in ["What is the capital of Italy?", "What is 7 * 6?"]:
        print(f"\n--- Query: {query} ---")
        selected = selector.select_examples({"question": query})
        print("Selected examples:", selected)
        rendered = few_shot.format(input=query)
        print(rendered)
        print("Model:", get_model().invoke(rendered).content)


if __name__ == "__main__":
    main()
