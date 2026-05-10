"""Memory utilities for conversation buffer and summary compression."""

from __future__ import annotations

from typing import Any

from agent.llm import build_chat_model

SUMMARY_PROMPT = (
    "Summarize the following conversation history into one concise paragraph "
    "preserving unresolved questions and user intent.\n\nHistory:\n{history}"
)
BUFFER_SIZE = 6


def _build_model():
    """Build OpenRouter chat model."""
    return build_chat_model(temperature=0, streaming=False)


def get_recent_buffer(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the last N conversation messages for active context."""
    return messages[-BUFFER_SIZE:]


def summarize_if_needed(state: dict[str, Any]) -> dict[str, Any]:
    """Compress older conversation turns into summary once threshold is exceeded."""
    try:
        if state.get("turn_count", 0) <= 5:
            return {"summary": state.get("summary", ""), "messages": get_recent_buffer(state.get("messages", []))}
        messages = state.get("messages", [])
        old_messages = messages[:-BUFFER_SIZE]
        history = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in old_messages])
        if not history.strip():
            return {"summary": state.get("summary", ""), "messages": get_recent_buffer(messages)}
        for attempt in range(1, 4):
            try:
                model = _build_model()
                summary_text = model.invoke(
                    SUMMARY_PROMPT.format(history=history)
                ).content.strip()
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 3:
                    import time
                    time.sleep(5 * attempt)
                else:
                    summary_text = state.get("summary", "")
                    break
        return {"summary": summary_text, "messages": get_recent_buffer(messages)}
    except Exception as exc:
        return {
            "summary": state.get("summary", ""),
            "messages": get_recent_buffer(state.get("messages", [])),
            "memory_error": str(exc),
        }
