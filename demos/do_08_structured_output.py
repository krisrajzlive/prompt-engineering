"""`with_structured_output`: the modern way to get typed output from a model.

Rather than hand-writing format instructions and parsing text yourself (see
demos/do_07_output_parsers.py), `with_structured_output(Schema)` uses the
provider's native structured-output / tool-calling support so the model is
constrained to return exactly that schema. Less prompt-engineering, more
reliability.
"""

from typing import Literal

from pydantic import BaseModel, Field

from core import get_model


class SupportTicket(BaseModel):
    """A triaged customer support ticket."""

    summary: str = Field(description="one-sentence summary of the issue")
    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    suggested_reply: str = Field(description="a short first reply to send the customer")


RAW_MESSAGE = (
    "Hi, I was charged twice for my subscription this month and I can't log in "
    "to check my invoices either. This is urgent, I need it fixed before my card "
    "gets charged again next week."
)


def main() -> None:
    model = get_model()
    structured_model = model.with_structured_output(SupportTicket)

    ticket = structured_model.invoke(
        f"Triage this customer message:\n\n{RAW_MESSAGE}"
    )
    print("--- Structured output (Pydantic schema via with_structured_output) ---")
    print(ticket)
    print("\nType:", type(ticket))
    print("Category:", ticket.category, "| Urgency:", ticket.urgency)

    # include_raw=True also returns the raw AIMessage + any parsing error,
    # useful when you want to inspect tool_calls or handle parse failures.
    verbose_model = model.with_structured_output(SupportTicket, include_raw=True)
    result = verbose_model.invoke(f"Triage this customer message:\n\n{RAW_MESSAGE}")
    print("\n--- include_raw=True ---")
    print("Keys:", list(result.keys()))
    print("Parsed:", result["parsed"])


if __name__ == "__main__":
    main()
