"""Prompt template basics: `PromptTemplate` vs `ChatPromptTemplate`.

`PromptTemplate` renders a single plain-text string (for completion-style
models). `ChatPromptTemplate` renders a *list of messages* (for chat
models) and is what nearly every modern LangChain app uses. Both support
`.format()` / `.format_messages()` for eager rendering and `.invoke()` for
use inside an LCEL chain (see demos/do_10_lcel_chains.py).
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


def main() -> None:
    # 1. PromptTemplate: single string, f-string style "{variable}" fills.
    text_prompt = PromptTemplate.from_template(
        "Write a one-sentence tagline for a {product} aimed at {audience}."
    )
    rendered = text_prompt.format(product="noise-cancelling headphones", audience="remote workers")
    print("--- PromptTemplate.format() ---")
    print(rendered)

    # 2. Explicit input_variables + template_format="jinja2" (loops/conditionals).
    jinja_prompt = PromptTemplate(
        template="Items: {% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}",
        input_variables=["items"],
        template_format="jinja2",
    )
    print("\n--- Jinja2 template_format ---")
    print(jinja_prompt.format(items=["mic", "camera", "ring light"]))

    # 3. ChatPromptTemplate: a list of (role, template) tuples -> list of messages.
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a terse product-naming assistant. Reply with one name only."),
            ("human", "Suggest a name for a {product} in the {theme} theme."),
        ]
    )
    messages = chat_prompt.format_messages(product="standing desk", theme="ocean")
    print("\n--- ChatPromptTemplate.format_messages() ---")
    for m in messages:
        print(f"[{m.type}] {m.content}")

    # 4. invoke() is the LCEL-native entry point: same result, Runnable interface.
    result = chat_prompt.invoke({"product": "standing desk", "theme": "ocean"})
    print("\n--- ChatPromptTemplate.invoke() -> PromptValue ---")
    print(result.to_messages())


if __name__ == "__main__":
    main()
