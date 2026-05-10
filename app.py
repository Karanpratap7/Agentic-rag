"""Streamlit app for the agentic arXiv RAG assistant."""

from __future__ import annotations

import json
import subprocess
import httpx
from pathlib import Path
from typing import Generator

import streamlit as st
from dotenv import load_dotenv

from agent.graph import build_graph
from agent.llm import build_chat_model

APP_TITLE = "ArXiv AI Research Assistant"
STREAM_PROMPT = (
    "You are an AI research assistant. Return the final user-facing answer based on this prepared draft. "
    "Preserve citations and factual content.\nDraft answer:\n{draft}"
)
BADGE_COLORS = {"retrieve": "green", "clarify": "orange", "refuse": "red", "tool": "blue", "answer_from_memory": "gray"}


def get_model():
    """Create streaming OpenRouter model."""
    return build_chat_model(temperature=0.2, streaming=True)


def stream_answer(draft: str) -> Generator[str, None, None]:
    """Yield token chunks from OpenRouter streaming API for UI rendering."""
    try:
        from agent.llm import PRIMARY_MODEL, OPENROUTER_BASE_URL
        import os
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": PRIMARY_MODEL,
            "messages": [{"role": "user", "content": STREAM_PROMPT.format(draft=draft)}],
            "stream": True
        }
        
        with httpx.Client() as client:
            with client.stream("POST", f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                content = data["choices"][0].get("delta", {}).get("content")
                                if content:
                                    yield content
                            
                            if "usage" in data and data["usage"]:
                                reasoning_tokens = data["usage"].get("reasoningTokens")
                                if reasoning_tokens:
                                    yield f"\n\n[Reasoning tokens: {reasoning_tokens}]"
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        for token in draft.split():
            yield token + " "


def init_session() -> None:
    """Initialize Streamlit session state containers."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("agent_state", {"messages": [], "summary": "", "trace": [], "turn_count": 0})
    st.session_state.setdefault("turn_count", 0)
    st.session_state.setdefault("last_trace", [])


def render_sidebar() -> None:
    """Render sidebar controls and metadata."""
    with st.sidebar:
        st.header("About")
        st.write("LangGraph agent with intent classification, query rewriting retrieval, arXiv tool use, and dual memory.")
        st.write(f"Current turn count: {st.session_state.turn_count}")
        with st.expander("View Last Trace"):
            st.json(st.session_state.last_trace if st.session_state.last_trace else {})
        if st.button("Run Evals"):
            result = subprocess.run(["python", "eval/run_eval.py"], capture_output=True, text=True, check=False)
            st.code(result.stdout or result.stderr)
            results_path = Path("eval/results.json")
            if results_path.exists():
                st.dataframe(json.loads(results_path.read_text(encoding="utf-8")))
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent_state = {"messages": [], "summary": "", "trace": [], "turn_count": 0}
            st.session_state.turn_count = 0
            st.session_state.last_trace = []


def render_message(role: str, content: str) -> None:
    """Render a single chat message with role placement."""
    with st.chat_message(role):
        st.markdown(content)


def main() -> None:
    """Run the Streamlit app loop."""
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    init_session()
    render_sidebar()
    graph = build_graph()
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"])
    user_query = st.chat_input("Ask about recent arXiv AI research...")
    if not user_query:
        return
    st.session_state.messages.append({"role": "user", "content": user_query})
    render_message("user", user_query)
    state = st.session_state.agent_state
    state["query"] = user_query
    state["messages"] = st.session_state.messages
    state["turn_count"] = st.session_state.turn_count + 1
    result = graph.invoke(state)
    st.session_state.turn_count = result.get("turn_count", state["turn_count"])
    st.session_state.agent_state = result
    st.session_state.last_trace = result.get("trace", [])
    with st.chat_message("assistant"):
        decision = result.get("decision", "unknown")
        color = BADGE_COLORS.get(decision, "gray")
        st.markdown(f":{color}[Intent: {decision}]")
        if result.get("rewritten_query"):
            with st.expander("Rewritten query"):
                st.write(result["rewritten_query"])
        draft = result.get("answer", "I do not have an answer yet.")
        try:
            streamed_text = st.write_stream(stream_answer(draft))
        except Exception as e:
            st.error(str(e))
            streamed_text = "Error generating response."
            
        if result.get("sources"):
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.markdown(f"- {source.get('title', 'Unknown')} — {source.get('url', '')}")
    st.session_state.messages.append({"role": "assistant", "content": streamed_text})
    st.session_state.agent_state["messages"] = st.session_state.messages
    st.session_state.agent_state["turn_count"] = st.session_state.turn_count


if __name__ == "__main__":
    main()
