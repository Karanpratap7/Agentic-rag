"""LangGraph nodes implementing agent decisions and actions."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.llm import build_chat_model
from agent.memory import summarize_if_needed
from agent.retrieval import retrieve_chunks
from agent.tools import search_arxiv

INTENT_PROMPT = (
    "You are an intent classifier for a research assistant. Given the user query and "
    "conversation history, classify the intent as exactly one of:\n"
    "- retrieve: question answerable from the arXiv corpus\n"
    "- tool: needs live arXiv search (asking about very recent or specific papers)\n"
    "- clarify: query is too ambiguous to retrieve effectively\n"
    "- refuse: query is out of domain (not about AI research) or harmful\n"
    "- answer_from_memory: answerable from conversation history alone\n"
    "Return ONLY the classification word, nothing else.\n"
    "Query: {query}\nHistory summary: {summary}"
)
CONTEXT_CHECK_PROMPT = (
    "Given query and retrieved evidence, output exactly one word: sufficient, contradictory, or empty.\n"
    "Query: {query}\nEvidence:\n{evidence}"
)
ANSWER_PROMPT = (
    "You are an AI research assistant. Answer using provided context with concise precision. "
    "If uncertain, state uncertainty. Include source titles inline.\n\n"
    "Summary memory: {summary}\n"
    "Recent messages:\n{messages}\n"
    "Retrieved docs:\n{docs}\n"
    "Tool result:\n{tool_result}\n"
    "User query: {query}"
)
CLARIFY_TEMPLATE = "Could you clarify your request with a specific AI research topic, method, or paper?"
REFUSE_TEMPLATE = "I can only help with AI research questions grounded in recent arXiv cs.AI literature."
TRACE_DIR = Path("traces")


def _model():
    """Construct OpenRouter model client."""
    return build_chat_model(temperature=0.2, streaming=False)


def _append_trace(state: dict[str, Any], node: str, decision: str, reasoning: str, in_data: Any, out_data: Any) -> None:
    """Append a structured trace record for observability."""
    item = {
        "node": node,
        "timestamp": datetime.now().isoformat(),
        "decision": decision,
        "reasoning": reasoning,
        "input_preview": str(in_data)[:100],
        "output_preview": str(out_data)[:100],
    }
    state.setdefault("trace", []).append(item)
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        trace_path = TRACE_DIR / f"trace-{datetime.now().strftime('%Y%m%d')}.jsonl"
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")
    except Exception:
        return


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    """Serialize recent conversation into prompt-safe text."""
    return "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages[-6:]])


def classify_intent(state: dict[str, Any]) -> dict[str, Any]:
    """Classify user intent to choose retrieval, tool, refusal, clarification, or memory answer path."""
    allowed = {"retrieve", "clarify", "refuse", "tool", "answer_from_memory"}
    query = state.get("query", "")

    def _normalize_label(raw_label: str) -> str:
        token = (raw_label or "").strip().lower().split()[0] if (raw_label or "").strip() else ""
        token = re.sub(r"[^a-z_]", "", token)
        return token

    try:
        raw = _model().invoke(INTENT_PROMPT.format(query=query, summary=state.get("summary", ""))).content
        decision = _normalize_label(raw)
        if decision not in allowed:
            # Heuristic fallback for robust routing when model output is malformed.
            lower_q = query.lower()
            if any(k in lower_q for k in ["latest", "today", "this week", "new paper", "arxiv id", "arxiv:"]):
                decision = "tool"
            elif lower_q.strip():
                decision = "retrieve"
            else:
                decision = "clarify"
    except Exception:
        # Heuristic fallback if model invocation fails (e.g., missing API key/transient outage).
        lower_q = query.lower()
        if any(k in lower_q for k in ["latest", "today", "this week", "new paper", "arxiv id", "arxiv:"]):
            decision = "tool"
        elif lower_q.strip():
            decision = "retrieve"
        else:
            decision = "clarify"
    _append_trace(state, "classify_intent", decision, "Intent gates graph routing for controllable behavior.", query, decision)
    return {"decision": decision}


# DECISION: retrieve step merged into rewrite_query_node because 
# retrieval is always preceded by rewriting — a separate node added 
# graph complexity without observability benefit.
def rewrite_query_node(state: dict[str, Any]) -> dict[str, Any]:
    """Rewrite and retrieve chunks for improved semantic recall."""
    rewritten, docs = retrieve_chunks(
        state.get("query", ""),
        state.setdefault("trace", []),
        rewrite_enabled=state.get("rewrite_enabled", True),
    )
    _append_trace(state, "rewrite_query", "rewritten", "Rewriting aligns user phrasing to technical corpus terminology.", state.get("query", ""), rewritten)
    return {"rewritten_query": rewritten, "retrieved_docs": docs}



def check_context(state: dict[str, Any]) -> dict[str, Any]:
    """Assess whether retrieved evidence is sufficient, contradictory, or empty."""
    docs = state.get("retrieved_docs", [])
    if not docs:
        decision = "empty"
    else:
        evidence = "\n".join([f"- {d.get('title', '')}: {d.get('chunk_text', '')[:240]}" for d in docs[:3]])
        try:
            raw = _model().invoke(CONTEXT_CHECK_PROMPT.format(query=state.get("query", ""), evidence=evidence)).content
            candidate = raw.strip().split()[0].lower()
            decision = candidate if candidate in {"sufficient", "contradictory", "empty"} else "sufficient"
        except Exception:
            decision = "sufficient"
    _append_trace(state, "check_context", decision, "Context quality controls answer/tool/clarification branch.", state.get("query", ""), decision)
    return {"context_decision": decision}


def call_arxiv_tool(state: dict[str, Any]) -> dict[str, Any]:
    """Call arXiv tool for recency- or metadata-sensitive queries."""
    result = search_arxiv(state.get("query", ""), max_results=5, trace=state.setdefault("trace", []))
    _append_trace(state, "call_arxiv_tool", "tool_called", "Tool use handles live-paper lookup beyond indexed corpus.", state.get("query", ""), result)
    return {"tool_result": result}


def _source_list(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build normalized source list for UI rendering and answer provenance."""
    sources: list[dict[str, str]] = []
    for d in docs:
        pid = d.get("paper_id", "")
        sources.append({"title": d.get("title", "Unknown"), "url": f"https://arxiv.org/abs/{pid}" if pid else ""})
    return sources


def generate_answer(state: dict[str, Any]) -> dict[str, Any]:
    """Generate final response using memory, retrieved context, and optional tool output."""
    try:
        docs = state.get("retrieved_docs", [])
        doc_text = "\n".join([f"{d.get('title', '')}: {d.get('chunk_text', '')[:500]}" for d in docs[:6]])
        answer = _model().invoke(
            ANSWER_PROMPT.format(
                summary=state.get("summary", ""),
                messages=_messages_to_text(state.get("messages", [])),
                docs=doc_text,
                tool_result=state.get("tool_result", ""),
                query=state.get("query", ""),
            )
        ).content.strip()
    except Exception as exc:
        answer = f"I encountered an internal error while composing the answer: {exc}"
    sources = _source_list(state.get("retrieved_docs", []))
    _append_trace(state, "generate_answer", "answered", "Final synthesis step grounded in available evidence and memory.", state.get("query", ""), answer)
    return {"answer": answer, "sources": sources}


def ask_clarification(state: dict[str, Any]) -> dict[str, Any]:
    """Return a clarification request when query lacks actionable specificity."""
    _append_trace(state, "ask_clarification", "clarify", "Clarification reduces retrieval failure from ambiguous prompts.", state.get("query", ""), CLARIFY_TEMPLATE)
    return {"answer": CLARIFY_TEMPLATE}


def refuse(state: dict[str, Any]) -> dict[str, Any]:
    """Refuse out-of-domain or unsafe requests while staying polite and scoped."""
    _append_trace(state, "refuse", "refused", "Domain-constrained refusal maintains assignment boundaries.", state.get("query", ""), REFUSE_TEMPLATE)
    return {"answer": REFUSE_TEMPLATE}


def update_memory(state: dict[str, Any]) -> dict[str, Any]:
    """Update buffer memory every turn and summary memory after threshold."""
    memory_update = summarize_if_needed(state)
    _append_trace(state, "update_memory", "memory_updated", "Combining recent buffer and compressed history balances context and token cost.", state.get("turn_count", 0), memory_update.get("summary", ""))
    return memory_update
