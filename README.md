# Prompt Engineering Lab

A LangChain project demonstrating prompt engineering mechanisms, templates,
input/output formats, and the different ways to call an LLM — using
OpenAI, Hugging Face Inference Providers, and Ollama Cloud as backends.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

`.env` was copied from `D:\workspace\langchain-llm-gateway\.env` and already
has `OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`, `OLLAMA_API_KEY`, and the
`LANGSMITH_*` keys filled in. See `.env.example` for the expected shape.

## Run

```bash
python main.py            # interactive menu
python main.py 6          # run a specific demo directly
```

## Demos

| # | File | Covers |
|---|------|--------|
| 1 | [demos/do_01_prompt_templates.py](demos/do_01_prompt_templates.py) | `PromptTemplate` vs `ChatPromptTemplate`, f-string vs jinja2 formats |
| 2 | [demos/do_02_message_types.py](demos/do_02_message_types.py) | `SystemMessage`/`HumanMessage`/`AIMessage`, `MessagesPlaceholder` for history |
| 3 | [demos/do_03_few_shot_prompting.py](demos/do_03_few_shot_prompting.py) | One-shot vs few-shot via `FewShotPromptTemplate`, `FewShotChatMessagePromptTemplate` |
| 4 | [demos/do_04_example_selectors.py](demos/do_04_example_selectors.py) | `SemanticSimilarityExampleSelector` + FAISS for dynamic few-shot |
| 5 | [demos/do_05_partial_and_composed_prompts.py](demos/do_05_partial_and_composed_prompts.py) | `.partial()` (static + callable), composing prompts from fragments |
| 6 | [demos/do_06_prompt_engineering_techniques.py](demos/do_06_prompt_engineering_techniques.py) | zero-shot vs few-shot vs chain-of-thought vs role-based prompting |
| 7 | [demos/do_07_output_parsers.py](demos/do_07_output_parsers.py) | `StrOutputParser`, list/JSON/Pydantic output parsers |
| 8 | [demos/do_08_structured_output.py](demos/do_08_structured_output.py) | `with_structured_output()` with a Pydantic schema |
| 9 | [demos/do_09_invocation_methods.py](demos/do_09_invocation_methods.py) | `invoke`/`batch`/`stream`/`ainvoke`/`abatch`/`astream` |
| 10 | [demos/do_10_lcel_chains.py](demos/do_10_lcel_chains.py) | LCEL `\|` composition, `RunnableLambda`/`Parallel`/`Passthrough` |
| 11 | [demos/do_11_multi_provider_prompting.py](demos/do_11_multi_provider_prompting.py) | Same prompt run against OpenAI, Hugging Face, and Ollama |
| 12 | [demos/do_12_langsmith_tracing.py](demos/do_12_langsmith_tracing.py) | LangSmith tracing with `@traceable` |
| 13 | [demos/do_13_tool_calling.py](demos/do_13_tool_calling.py) | `@tool`, `bind_tools`, reading `tool_calls`, feeding results back via `ToolMessage` |
| 14 | [demos/do_14_simple_rag.py](demos/do_14_simple_rag.py) | Minimal RAG: FAISS retriever + `RunnableParallel` in an LCEL chain |
| 15 | [demos/do_15_response_anatomy.py](demos/do_15_response_anatomy.py) | What's inside an `AIMessage`: content, usage_metadata, tool_calls, response_metadata, streaming chunks |
| 16 | [demos/do_16_prompt_chaining.py](demos/do_16_prompt_chaining.py) | Prompt chaining: extract → draft → tighten across 3 sequential LLM calls, manual + LCEL-piped |

## Layout

```
core/models.py   # get_model() / get_embeddings() — provider-agnostic init
demos/           # one focused script per concept, each with a main()
main.py          # menu/CLI runner
```
