"""Message types and `MessagesPlaceholder`: building real chat turns.

Chat models don't take a single string; they take a list of typed messages
(`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`). A
`MessagesPlaceholder` reserves a slot in a `ChatPromptTemplate` for a whole
list of prior turns, which is how conversation history / memory gets
spliced into a prompt at call time.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core import get_model


def main() -> None:
    # 1. Raw message objects, no template needed.
    messages = [
        SystemMessage(content="You are a concise Python tutor."),
        HumanMessage(content="What does `zip()` do?"),
    ]
    model = get_model()
    response = model.invoke(messages)
    print("--- Raw message list -> model.invoke() ---")
    print(response.content)

    # 2. Simulate a running conversation and feed it back in via
    #    MessagesPlaceholder, so history is a first-class template slot.
    history = [
        HumanMessage(content="My favorite language is Python."),
        AIMessage(content="Noted — Python it is."),
    ]
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Keep answers to one sentence."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    rendered = chat_prompt.invoke(
        {"history": history, "question": "What's my favorite language?"}
    )
    print("\n--- Rendered messages with history spliced in ---")
    for m in rendered.to_messages():
        print(f"[{m.type}] {m.content}")

    response = model.invoke(rendered)
    print("\n--- Model response (uses the injected history) ---")
    print(response.content)


if __name__ == "__main__":
    main()
