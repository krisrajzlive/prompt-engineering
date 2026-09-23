"""CLI menu to run any prompt-engineering demo.

Usage:
    python main.py            # interactive menu
    python main.py 3          # run demo 3 directly
"""

import importlib
import sys

DEMOS = {
    "1": ("Prompt templates: PromptTemplate vs ChatPromptTemplate", "demos.do_01_prompt_templates"),
    "2": ("Message types + MessagesPlaceholder (chat history)", "demos.do_02_message_types"),
    "3": ("One-shot vs few-shot prompting (text + chat style)", "demos.do_03_few_shot_prompting"),
    "4": ("Example selectors (semantic similarity)", "demos.do_04_example_selectors"),
    "5": ("Partial variables + composed prompts", "demos.do_05_partial_and_composed_prompts"),
    "6": ("Zero-shot vs few-shot vs CoT vs role-based prompting", "demos.do_06_prompt_engineering_techniques"),
    "7": ("Output parsers (str/list/json/pydantic)", "demos.do_07_output_parsers"),
    "8": ("Structured output via with_structured_output", "demos.do_08_structured_output"),
    "9": ("Invocation methods: invoke/batch/stream/async", "demos.do_09_invocation_methods"),
    "10": ("LCEL chains: RunnableLambda/Parallel/Passthrough", "demos.do_10_lcel_chains"),
    "11": ("Prompt portability across OpenAI/HuggingFace/Ollama", "demos.do_11_multi_provider_prompting"),
    "12": ("LangSmith tracing", "demos.do_12_langsmith_tracing"),
    "13": ("Tool calling: bind_tools + tool_calls loop", "demos.do_13_tool_calling"),
    "14": ("Simple RAG: retriever + LCEL chain", "demos.do_14_simple_rag"),
    "15": ("Response anatomy: AIMessage fields (content/usage/tool_calls/...)", "demos.do_15_response_anatomy"),
}


def run(choice: str) -> None:
    if choice not in DEMOS:
        print(f"Unknown demo '{choice}'. Choose from: {', '.join(DEMOS)}")
        sys.exit(1)
    label, module_name = DEMOS[choice]
    print(f"\n=== Demo {choice}: {label} ===\n")
    module = importlib.import_module(module_name)
    module.main()


def menu() -> None:
    print("Prompt Engineering Lab demos:")
    for key, (label, _) in DEMOS.items():
        print(f"  {key}. {label}")
    choice = input("Pick a demo number: ").strip()
    run(choice)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        menu()
