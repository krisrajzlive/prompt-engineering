"""LCEL (LangChain Expression Language): composing prompts with `|`.

`prompt | model | parser` is itself a `Runnable`, built by piping smaller
Runnables together. 
`RunnablePassthrough` forwards a value unchanged (handy
for keeping the original input alongside a transformed one),
`RunnableLambda` wraps a plain Python function as a pipeline step, and
`RunnableParallel` fans one input out to several branches at once.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from core import get_model


def main() -> None:
    model = get_model()
    parser = StrOutputParser()

    # 1. The basic three-stage chain.
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize this text in one sentence:\n{text}"
    )
    summarize_chain = summarize_prompt | model | parser

    text = (
        "LangChain's Expression Language lets you compose Runnables with the "
        "`|` operator, similar to a Unix pipe, so prompts, models and parsers "
        "become interchangeable, testable building blocks."
    )
    print("--- Basic chain ---")
    print(summarize_chain.invoke({"text": text}))

    # 2. RunnableLambda: a plain function as a pipeline step (pre/post-processing).
    uppercase_step = RunnableLambda(lambda s: s.upper())
    shouting_chain = summarize_chain | uppercase_step
    print("\n--- Chain + RunnableLambda post-processing ---")
    print(shouting_chain.invoke({"text": text}))

    # 3. RunnablePassthrough: carry the original input alongside a derived value,
    #    e.g. returning both the summary and the original text length.
    with_metadata = RunnableParallel(
        summary=summarize_chain,
        original_length=RunnableLambda(lambda x: len(x["text"])),
        original=RunnablePassthrough(),
    )
    print("\n--- RunnableParallel (summary + metadata + passthrough) ---")
    result = with_metadata.invoke({"text": text})
    print("Summary:", result["summary"])
    print("Original length:", result["original_length"])
    print("Passthrough echoes input dict:", result["original"].keys())

    # 4. Two independent prompts run in parallel against the same input.
    translate_prompt = ChatPromptTemplate.from_template("Translate to French:\n{text}")
    translate_chain = translate_prompt | model | parser

    dual_chain = RunnableParallel(summary=summarize_chain, french=translate_chain)
    print("\n--- Two chains fanned out in parallel ---")
    result = dual_chain.invoke({"text": text})
    print("Summary:", result["summary"])
    print("French:", result["french"])


if __name__ == "__main__":
    main()
