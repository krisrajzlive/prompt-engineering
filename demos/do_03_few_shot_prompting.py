"""Few-shot prompting: `FewShotPromptTemplate` and its chat equivalent.

Showing the model a handful of worked (input, output) examples before the
real question steers its output format far more reliably than instructions
alone. LangChain has a dedicated template for this so the example list stays
data (easy to edit/extend) rather than baked into a hand-written string.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
)

from core import get_model

EXAMPLES = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "fast", "antonym": "slow"},
]


def main() -> None:
    model = get_model()

    # 1. Text-completion style few-shot prompt.
    example_prompt = PromptTemplate.from_template("Word: {word}\nAntonym: {antonym}")
    few_shot = FewShotPromptTemplate(
        examples=EXAMPLES,
        example_prompt=example_prompt,
        prefix="Give the antonym of each word.",
        suffix="Word: {input}\nAntonym:",
        input_variables=["input"],
    )
    rendered = few_shot.format(input="ancient")
    print("--- FewShotPromptTemplate (text) ---")
    print(rendered)
    print("\nModel:", model.invoke(rendered).content)

    # 2. Chat-style few-shot: examples become alternating human/ai turns,
    #    which chat models tend to follow more reliably than a text blob.
    example_chat_prompt = ChatPromptTemplate.from_messages(
        [("human", "{word}"), ("ai", "{antonym}")]
    )
    few_shot_chat = FewShotChatMessagePromptTemplate(
        examples=EXAMPLES,
        example_prompt=example_chat_prompt,
    )
    full_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Reply with a single-word antonym, nothing else."),
            few_shot_chat,
            ("human", "{word}"),
        ]
    )
    messages = full_chat_prompt.invoke({"word": "ancient"})
    print("\n--- FewShotChatMessagePromptTemplate rendered turns ---")
    for m in messages.to_messages():
        print(f"[{m.type}] {m.content}")

    response = model.invoke(messages)
    print("\nModel:", response.content)


if __name__ == "__main__":
    main()
