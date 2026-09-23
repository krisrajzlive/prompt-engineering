"""One-shot and few-shot prompting: `FewShotPromptTemplate` and its chat equivalent.

Showing the model worked (input, output) examples before the real question
steers its output format far more reliably than instructions alone.
"One-shot" is just the k=1 special case of few-shot — a single example to
pin down the format — while "few-shot" uses several to also pin down the
*pattern* across varied inputs. LangChain doesn't have a separate one-shot
template; it's the same `FewShotPromptTemplate` with a one-item example
list, which is why it's demoed first here before scaling up to several
examples. The example list stays data (easy to edit/extend) rather than
baked into a hand-written string either way.
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

    # 1. One-shot: a single worked example, just enough to pin down the
    #    output format (one word, lowercase, no extra text) without yet
    #    showing any variety of input.
    example_prompt = PromptTemplate.from_template("Word: {word}\nAntonym: {antonym}")
    one_shot = FewShotPromptTemplate(
        examples=EXAMPLES[:1],
        example_prompt=example_prompt,
        prefix="Give the antonym of each word.",
        suffix="Word: {input}\nAntonym:",
        input_variables=["input"],
    )
    rendered = one_shot.format(input="ancient")
    print("--- FewShotPromptTemplate, one example (one-shot) ---")
    print(rendered)
    print("\nModel:", model.invoke(rendered).content)

    # 2. Few-shot: the same template, but with several examples so the
    #    model also sees the *pattern* holds across varied inputs, not
    #    just the format of one.
    few_shot = FewShotPromptTemplate(
        examples=EXAMPLES,
        example_prompt=example_prompt,
        prefix="Give the antonym of each word.",
        suffix="Word: {input}\nAntonym:",
        input_variables=["input"],
    )
    rendered = few_shot.format(input="ancient")
    print("\n--- FewShotPromptTemplate, several examples (few-shot) ---")
    print(rendered)
    print("\nModel:", model.invoke(rendered).content)

    # 3. Chat-style few-shot: examples become alternating human/ai turns,
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
