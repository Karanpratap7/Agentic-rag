"""Shared LLM client factory configured for OpenRouter."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")


def build_chat_model(*, temperature: float = 0.2, streaming: bool = False, is_fallback: bool = False) -> ChatOpenAI:
    """Create OpenRouter chat model with env-loaded API key and fallback support."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model_name = FALLBACK_MODEL if is_fallback else PRIMARY_MODEL
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        streaming=streaming,
    )
