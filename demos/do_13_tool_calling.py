"""Tool calling: `bind_tools`, `tool_calls`, and feeding results back.

`@tool` turns a plain Python function into a LangChain `Tool` (itself a
`Runnable`) with a name/description/args schema derived from its signature
and docstring. `model.bind_tools([...])` doesn't execute anything — it just
tells the model which tools it *may* call. The model then decides whether
to call one, and the caller is responsible for actually running it and
feeding the result back as a `ToolMessage` for a final natural-language
answer.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from core import get_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny and 22C in {city}."


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using a fixed demo rate."""
    rates = {("USD", "INR"): 83.0, ("USD", "EUR"): 0.92, ("EUR", "INR"): 90.0}
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate is None:
        return f"No rate available for {from_currency} -> {to_currency}."
    return f"{amount} {from_currency} = {amount * rate:.2f} {to_currency}"


TOOLS = [get_weather, convert_currency]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def main() -> None:
    model = get_model().bind_tools(TOOLS)

    # 1. A question that should trigger exactly one tool call.
    print("--- Single tool call ---")
    messages = [HumanMessage("What's the weather like in Chennai?")]
    ai_message = model.invoke(messages)
    print("Tool calls requested:", ai_message.tool_calls)
    messages.append(ai_message)

    for call in ai_message.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        result = tool_fn.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = model.invoke(messages)
    print("Final answer:", final.content)

    # 2. A question that should trigger a *different* tool, from the same bound set.
    print("\n--- Model picks the right tool among several ---")
    messages = [HumanMessage("Convert 100 USD to INR.")]
    ai_message = model.invoke(messages)
    print("Tool calls requested:", ai_message.tool_calls)
    messages.append(ai_message)

    for call in ai_message.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        result = tool_fn.invoke(call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = model.invoke(messages)
    print("Final answer:", final.content)

    # 3. A question with no matching tool: the model should just answer directly.
    print("\n--- No tool needed ---")
    ai_message = model.invoke([HumanMessage("What is 12 * 12?")])
    print("Tool calls requested:", ai_message.tool_calls)
    print("Direct answer:", ai_message.content)


if __name__ == "__main__":
    main()
