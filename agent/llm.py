"""Shared LLM client factory configured for OpenRouter."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRIMARY_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "google/gemma-2-9b-it:free")


def build_chat_model(*, temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    """Create OpenRouter chat model with env-loaded API key and fallback."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    try:
        return ChatOpenAI(
            model=PRIMARY_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=temperature,
            streaming=streaming,
        )
    except Exception as exc:
        print(f"[llm] Primary model failed ({exc}), falling back to {FALLBACK_MODEL}")
        return ChatOpenAI(
            model=FALLBACK_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=temperature,
            streaming=streaming,
        )
