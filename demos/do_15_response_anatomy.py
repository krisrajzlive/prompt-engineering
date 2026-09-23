"""Anatomy of a model response: what's actually inside the `AIMessage`.

`model.invoke(...)` doesn't just return text — it returns an `AIMessage`
object with several fields worth knowing about, most of which people never
look at because `.content` is all they print. This walks through the ones
that matter in practice: the answer itself, token usage/cost accounting,
why generation stopped, the provider's raw response metadata, and (when
tools are bound) any tool calls the model wants executed.
"""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from core import get_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny and 22C in {city}."


def inspect(response: AIMessage, label: str) -> None:
    print(f"\n--- {label} ---")
    print("type(response):        ", type(response).__name__)

    # 1. content: the actual text answer. Can be "" when the model only
    #    returned tool calls, so never assume it's non-empty.
    print("content:                ", repr(response.content))

    # 2. id: a unique id for this generation, useful for tracing/logging.
    print("id:                     ", response.id)

    # 3. response_metadata: provider-specific info about the call — model
    #    name actually used, why generation stopped, and (for OpenAI) the
    #    raw token usage block.
    meta = response.response_metadata
    print("response_metadata keys: ", list(meta.keys()))
    print("  model_name:           ", meta.get("model_name"))
    print("  finish_reason:        ", meta.get("finish_reason"))

    # 4. usage_metadata: a normalized, provider-agnostic token count —
    #    prefer this over digging through response_metadata for cost/usage
    #    tracking, since its shape is the same across OpenAI/Anthropic/etc.
    print("usage_metadata:         ", response.usage_metadata)

    # 5. tool_calls: populated only when the model decided to call a tool
    #    bound via bind_tools() (see demos/do_13_tool_calling.py). Empty
    #    list, not None, when no tool call was made.
    print("tool_calls:             ", response.tool_calls)

    # 6. additional_kwargs: provider-specific extras that didn't get a
    #    normalized field of their own (e.g. raw OpenAI function_call/refusal).
    print("additional_kwargs keys: ", list(response.additional_kwargs.keys()))


def main() -> None:
    model = get_model()

    # A plain text response.
    response = model.invoke("In one sentence, what is the speed of light?")
    inspect(response, "Plain text response")

    # A response where the model calls a tool instead of answering directly —
    # note content is empty and tool_calls is populated.
    tool_model = model.bind_tools([get_weather])
    tool_response = tool_model.invoke("What's the weather in Chennai?")
    inspect(tool_response, "Tool-calling response")

    # Streaming: each chunk is an AIMessageChunk, a lighter-weight subclass
    # of AIMessage. Chunks accumulate with `+`; usage/finish_reason usually
    # only appear fully populated on the final chunk.
    print("\n--- Streaming chunks (AIMessageChunk) ---")
    final_chunk = None
    for chunk in model.stream("Say hi in three words."):
        print(f"  chunk: content={chunk.content!r} usage_metadata={chunk.usage_metadata}")
        final_chunk = chunk if final_chunk is None else final_chunk + chunk
    print("Accumulated content:   ", repr(final_chunk.content))
    print("Accumulated usage:     ", final_chunk.usage_metadata)


if __name__ == "__main__":
    main()
