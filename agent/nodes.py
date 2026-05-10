"""LangGraph nodes implementing agent decisions and actions."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

SESSION_ID = datetime.now().strftime('%Y%m%d-%H%M%S')

from agent.llm import build_chat_model
from agent.memory import summarize_if_needed
from agent.retrieval import retrieve_chunks
from agent.tools import search_arxiv

MIN_CALL_INTERVAL_SECONDS = 2.0  # Minimum gap between LLM API calls on free tier
_last_call_time: float = 0.0

import time

def _invoke_with_retry(model, prompt: str, max_attempts: int = 3) -> str:
    """Invoke LLM with exponential backoff on rate limit errors.
    
    # DECISION: Free-tier OpenRouter models have aggressive rate limits.
    # Retry with backoff prevents 429 failures from crashing the agent
    # graph mid-execution.
    """
    global _last_call_time
    import time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL_SECONDS:
        time.sleep(MIN_CALL_INTERVAL_SECONDS - elapsed)
    _last_call_time = time.time()

    for attempt in range(1, max_attempts + 1):
        try:
            return model.invoke(prompt).content
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and attempt < max_attempts:
                wait = 5 * attempt  # 5s, 10s
                print(f"[llm] Rate limited (attempt {attempt}), "
                      f"waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise
    return ""

INTENT_PROMPT = (
    "You are an intent classifier for a research assistant that has access to "
    "a local corpus of arXiv cs.AI papers from the last 90 days.\n\n"
    "Classify the user query as exactly one of:\n"
    "- retrieve: question about AI research concepts, methods, or findings "
    "that can be answered from the local paper corpus. Also use this for "
    "questions referencing 'this dataset', 'these papers', 'the corpus', "
    "'trends you see', or any question about what is IN the indexed collection.\n"
    "- tool: needs a LIVE arXiv search — only use when the query explicitly "
    "asks for papers published TODAY, THIS WEEK, by a SPECIFIC AUTHOR by name, "
    "or by a specific arXiv paper ID.\n"
    "- clarify: query is too vague to retrieve effectively (e.g. 'tell me about it', "
    "'what do you think about AI?')\n"
    "- refuse: query is completely unrelated to AI research (cooking, sports, etc.)\n"
    "- answer_from_memory: answerable from conversation history alone "
    "(e.g. 'summarize what we discussed', 'what did I ask earlier?')\n\n"
    "When in doubt between retrieve and tool, prefer retrieve.\n\n"
    "Return ONLY the classification word, nothing else.\n"
    "Query: {query}\nHistory summary: {summary}"
)
CONTEXT_CHECK_PROMPT = (
    "You are evaluating retrieved evidence for a RAG system.\n\n"
    "Given the user query and retrieved evidence chunks, output exactly "
    "one word from: sufficient, contradictory, or empty.\n\n"
    "Rules:\n"
    "- sufficient: the retrieved chunks contain relevant information that "
    "can help answer the query, even if they cover different aspects\n"
    "- contradictory: the retrieved chunks make directly opposing factual "
    "claims about the SAME specific topic (e.g. one says method X outperforms "
    "Y, another says Y outperforms X)\n"
    "- empty: the retrieved chunks are completely irrelevant to the query "
    "or no chunks were retrieved\n\n"
    "Important: diverse chunks covering different subtopics is NOT "
    "contradictory — it is sufficient.\n\n"
    "Query: {query}\n"
    "Evidence:\n{evidence}"
)
ANSWER_PROMPT = (
    "You are an AI research assistant with access to a corpus of recent "
    "arXiv cs.AI papers.\n\n"
    "Instructions:\n"
    "- For trend or summary questions: identify patterns across multiple "
    "retrieved papers, group by theme, and name specific papers as examples\n"
    "- For specific technical questions: answer precisely using the most "
    "relevant retrieved chunks\n"
    "- Always cite paper titles inline when making specific claims\n"
    "- If context is insufficient, say so explicitly rather than fabricating\n"
    "- Be concise but substantive\n\n"
    "Summary memory: {summary}\n"
    "Recent messages:\n{messages}\n\n"
    "Retrieved papers and chunks:\n{docs}\n\n"
    "Tool result (live arXiv search):\n{tool_result}\n\n"
    "User query: {query}\n\n"
    "Answer:"
)
CLARIFY_TEMPLATE = "Could you clarify your request with a specific AI research topic, method, or paper?"
REFUSE_TEMPLATE = "I can only help with AI research questions grounded in recent arXiv cs.AI literature."
TRACE_DIR = Path("traces")


def _model(is_fallback: bool = False):
    """Construct OpenRouter model client."""
    return build_chat_model(temperature=0.2, streaming=False, is_fallback=is_fallback)


def _invoke_with_retry(model_factory, prompt: str, max_attempts: int = 3) -> str:
    """Invoke LLM with exponential backoff on rate limit errors and dynamic fallback.
    
    # DECISION: Free-tier OpenRouter models have aggressive rate limits.
    # Retry with backoff prevents 429 failures from crashing the agent
    # graph mid-execution. If 429s persist or a 500 occurs, swaps to fallback model.
    """
    global _last_call_time
    import time

    # Try primary model
    model = model_factory(is_fallback=False)
    for attempt in range(1, max_attempts + 1):
        elapsed = time.time() - _last_call_time
        if elapsed < MIN_CALL_INTERVAL_SECONDS:
            time.sleep(MIN_CALL_INTERVAL_SECONDS - elapsed)
        _last_call_time = time.time()

        try:
            return model.invoke(prompt).content
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and attempt < max_attempts:
                wait = 5 * attempt  # 5s, 10s
                print(f"[llm] Rate limited (attempt {attempt}), "
                      f"waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"[llm] Primary model failed ({err_str}), swapping to fallback model...")
                break

    # Try fallback model
    fallback_model = model_factory(is_fallback=True)
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL_SECONDS:
        time.sleep(MIN_CALL_INTERVAL_SECONDS - elapsed)
    _last_call_time = time.time()

    try:
        return fallback_model.invoke(prompt).content
    except Exception as exc:
        print(f"[llm] Fallback model also failed: {exc}")
        raise


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
        trace_path = TRACE_DIR / f"trace-{SESSION_ID}.jsonl"
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
        raw = _invoke_with_retry(
            _model,
            INTENT_PROMPT.format(query=query, summary=state.get("summary", ""))
        )
        decision = _normalize_label(raw)
        if decision not in allowed:
            # Heuristic fallback for robust routing when model output is malformed.
            lower_q = query.lower()
            if any(k in lower_q for k in ["today", "this week", "last week", "arxiv id", "arxiv:", "by author"]):
                decision = "tool"
            elif lower_q.strip():
                decision = "retrieve"
            else:
                decision = "clarify"
    except Exception:
        # Heuristic fallback if model invocation fails (e.g., missing API key/transient outage).
        lower_q = query.lower()
        if any(k in lower_q for k in ["today", "this week", "last week", "arxiv id", "arxiv:", "by author"]):
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
            raw = _invoke_with_retry(
                _model,
                CONTEXT_CHECK_PROMPT.format(
                    query=state.get("query", ""), evidence=evidence
                )
            )
            candidate = raw.strip().split()[0].lower()
            decision = candidate if candidate in {"sufficient", "contradictory", "empty"} else "sufficient"
        except Exception:
            decision = "sufficient"
    _append_trace(state, "check_context", decision, "Context quality controls answer/tool/clarification branch.", state.get("query", ""), decision)
    return {"context_decision": decision, "context_contradictory": decision == "contradictory"}


def call_arxiv_tool(state: dict[str, Any]) -> dict[str, Any]:
    """Call arXiv tool for recency/metadata queries; fall back to retrieval on failure."""
    result = search_arxiv(
        state.get("query", ""), max_results=5, trace=state.setdefault("trace", [])
    )
    if result.startswith("arXiv tool error:"):
        # DECISION: On tool failure, fall back to corpus retrieval rather than 
        # surfacing an API error to the user. Degraded retrieval is better than 
        # an error message as context for generation.
        _append_trace(
            state, "call_arxiv_tool", "tool_failed_fallback",
            "arXiv API failed; falling back to FAISS corpus retrieval.",
            state.get("query", ""), "fallback_to_retrieve"
        )
        rewritten, docs = retrieve_chunks(
            state.get("query", ""),
            state.setdefault("trace", []),
            rewrite_enabled=state.get("rewrite_enabled", True),
        )
        return {
            "tool_result": "arXiv API unavailable. Answer based on indexed corpus.",
            "retrieved_docs": docs,
            "rewritten_query": rewritten,
        }
    _append_trace(
        state, "call_arxiv_tool", "tool_called",
        "Tool use handles live-paper lookup beyond indexed corpus.",
        state.get("query", ""), result
    )
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
        answer = _invoke_with_retry(
            _model,
            ANSWER_PROMPT.format(
                summary=state.get("summary", ""),
                messages=_messages_to_text(state.get("messages", [])),
                docs=doc_text,
                tool_result=state.get("tool_result", ""),
                query=state.get("query", ""),
            )
        ).strip()
    except Exception as exc:
        answer = f"I encountered an internal error while composing the answer: {exc}"
        
    if state.get("context_contradictory"):
        answer = "⚠️ Note: Retrieved sources contain conflicting information on this topic.\n\n" + answer
        
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
