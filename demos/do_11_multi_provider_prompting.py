"""Prompt portability: the same `ChatPromptTemplate` across three providers.

Because `ChatPromptTemplate` renders provider-agnostic LangChain message
objects, and `init_chat_model` returns a uniform `BaseChatModel` regardless
of vendor, one prompt can be pointed at OpenAI, Hugging Face Inference
Providers, or an Ollama-hosted open model without touching the prompt
itself — only the model config changes.
"""

from langchain_core.prompts import ChatPromptTemplate

from core import HUGGINGFACE_KWARGS, OLLAMA_KWARGS, get_model

PROVIDERS = [
    ("openai:gpt-4o-mini", {}),
    ("meta-llama/Llama-3.1-8B-Instruct", HUGGINGFACE_KWARGS),
    ("gpt-oss:20b", OLLAMA_KWARGS),
]

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a precise technical writer. Answer in exactly two sentences."),
        ("human", "Explain what {concept} is and why it matters for {audience}."),
    ]
)


def main() -> None:
    inputs = {"concept": "prompt engineering", "audience": "backend engineers"}
    for model_id, kwargs in PROVIDERS:
        model = get_model(model_id, **kwargs)
        chain = PROMPT | model
        try:
            response = chain.invoke(inputs)
            print(f"--- {model_id} ---")
            print(response.content)
            print()
        except Exception as exc:  # provider outages/credit limits shouldn't kill the demo
            print(f"--- {model_id} ---")
            print(f"[skipped: {exc}]\n")


if __name__ == "__main__":
    main()
