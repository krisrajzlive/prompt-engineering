"""Provider-agnostic model/embeddings init, shared by every demo.

`init_chat_model` is LangChain's uniform entry point: pass a "provider:model"
string (or a bare model id plus `model_provider`/`base_url`/`api_key`) and
get back a `BaseChatModel`. That's what lets the same prompt template be fed
to OpenAI, Hugging Face or Ollama-hosted models interchangeably in these
demos (see demos/do_11_multi_provider_prompting.py).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

try:
    # Windows consoles default to a legacy codepage (e.g. cp1252) that
    # can't render every character an LLM returns (smart quotes, en/em
    # dashes, etc.), crashing print() with a UnicodeEncodeError.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

DEFAULT_MODEL = "openai:gpt-4o-mini"

# Ready-made kwargs for the two OpenAI-compatible free-tier gateways used
# across these demos.
HUGGINGFACE_KWARGS = {
    "model_provider": "openai",
    "base_url": "https://router.huggingface.co/v1",
    "api_key": os.environ.get("HUGGINGFACE_API_KEY"),
}
OLLAMA_KWARGS = {
    "model_provider": "openai",
    "base_url": "https://ollama.com/v1",
    "api_key": os.environ.get("OLLAMA_API_KEY"),
}


def get_model(model: str = DEFAULT_MODEL, **kwargs) -> BaseChatModel:
    """Return a chat model instance for the given "provider:model" string."""
    return init_chat_model(model, **kwargs)


def get_embeddings() -> Embeddings:
    """Return an embeddings model (Hugging Face Inference API, free tier)."""
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACE_API_KEY"),
    )
