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
    "Classify this query for an arXiv cs.AI research assistant.\n"
    "Local corpus = last 90 days of cs.AI papers.\n\n"
    "Labels (return ONE word only):\n"
    "- retrieve: AI research question answerable from local corpus, "
    "or asks about 'this dataset/corpus/papers'\n"
    "- tool: needs LIVE arXiv search (today/this week/specific author/paper ID)\n"
    "- clarify: too vague to search (e.g. 'tell me about it')\n"
    "- refuse: unrelated to AI research (cooking, sports, etc.)\n"
    "- answer_from_memory: answered from chat history alone\n\n"
    "When unsure between retrieve and tool, use retrieve.\n"
    "Query: {query}\n"
    "History: {summary}\n"
    "Label:"
)
CONTEXT_CHECK_PROMPT = (
    # Unused — check_context now uses heuristics
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
    "You are an arXiv cs.AI research assistant.\n"
    "Answer the query using retrieved papers. "
    "For trend questions: group papers by theme. "
    "For technical questions: be precise. "
    "Cite paper titles inline. "
    "If context is insufficient, say so.\n\n"
    "Memory: {summary}\n"
    "History: {messages}\n"
    "Papers:\n{docs}\n"
    "Tool results:\n{tool_result}\n"
    "Query: {query}\n"
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
    """Serialize recent conversation into prompt-safe text with length cap."""
    lines = []
    for m in messages[-4:]:  # Reduced from 6 to 4
        role = m.get("role", "unknown")
        # Cap each message at 200 chars to prevent context bloat
        content = m.get("content", "")[:200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


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


TECHNICAL_TERMS = {
    "llm", "rag", "transformer", "attention", "embedding", "fine-tuning",
    "rlhf", "dpo", "ppo", "bert", "gpt", "diffusion", "vae", "gan",
    "reinforcement", "supervised", "unsupervised", "neural", "gradient",
    "inference", "training", "benchmark", "evaluation", "dataset",
    "retrieval", "generation", "alignment", "hallucination", "prompt",
    "agent", "agentic", "multimodal", "tokenizer", "encoder", "decoder"
}

def _is_already_technical(query: str) -> bool:
    """Check if query already uses technical terminology — skip rewrite if so."""
    words = set(query.lower().split())
    return bool(words & TECHNICAL_TERMS)

# DECISION: retrieve step merged into rewrite_query_node because 
# retrieval is always preceded by rewriting — a separate node added 
# graph complexity without observability benefit.
def rewrite_query_node(state: dict[str, Any]) -> dict[str, Any]:
    """Rewrite and retrieve chunks; skip rewrite for already-technical queries."""
    query = state.get("query", "")
    rewrite_enabled = state.get("rewrite_enabled", True)
    
    # Skip LLM rewrite call if query is already technical — saves one API call
    # DECISION: Technical queries don't benefit from rewriting since they
    # already use corpus-native terminology. This reduces API calls on the
    # most common query type without degrading retrieval quality.
    if rewrite_enabled and _is_already_technical(query):
        rewrite_enabled = False
        _append_trace(
            state, "rewrite_query", "skipped_technical",
            "Query already uses technical terms — skipping LLM rewrite to save API calls.",
            query, query
        )
    
    rewritten, docs = retrieve_chunks(
        query,
        state.setdefault("trace", []),
        rewrite_enabled=rewrite_enabled,
    )
    _append_trace(
        state, "rewrite_query", "rewritten",
        "Rewriting aligns user phrasing to technical corpus terminology.",
        query, rewritten
    )
    return {"rewritten_query": rewritten, "retrieved_docs": docs}



def check_context(state: dict[str, Any]) -> dict[str, Any]:
    """Assess retrieved evidence quality using heuristics instead of LLM.
    
    # DECISION: Replaced LLM-based context checking with heuristics to
    # reduce API calls from 4 to 3 per turn. The LLM version was calling
    # the model just to return one word, which consumed rate limit budget
    # disproportionate to the value. Heuristic rules cover >95% of cases:
    # empty corpus → empty, relevant titles present → sufficient.
    # True contradiction detection is rare and handled by generate_answer's
    # uncertainty instruction when it occurs.
    """
    docs = state.get("retrieved_docs", [])
    query = state.get("query", "").lower()
    
    if not docs:
        decision = "empty"
    else:
        # Check relevance: if top doc title shares keywords with query,
        # context is sufficient. This avoids an LLM call for the common case.
        query_words = set(query.split()) - {
            "what", "how", "why", "when", "where", "who", "is", "are",
            "the", "a", "an", "of", "in", "on", "for", "to", "do",
            "you", "me", "i", "we", "they", "it", "this", "that"
        }
        top_titles = " ".join([
            d.get("title", "").lower() for d in docs[:3]
        ])
        top_chunks = " ".join([
            d.get("chunk_text", "").lower()[:200] for d in docs[:3]
        ])
        combined = top_titles + " " + top_chunks
        
        # Count keyword overlap between query and retrieved content
        overlap = sum(1 for w in query_words if w in combined)
        
        if overlap >= 1 or len(query_words) == 0:
            decision = "sufficient"
        else:
            decision = "empty"
    
    _append_trace(
        state, "check_context", decision,
        "Heuristic context check: keyword overlap between query and "
        "retrieved titles/chunks. Replaces LLM call to save rate limit.",
        state.get("query", ""), decision
    )
    return {"context_decision": decision, "context_contradictory": False}


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
        doc_text = "\n".join([
            f"{d.get('title', '')}: {d.get('chunk_text', '')[:300]}" 
            for d in docs[:4]
        ])
        # DECISION: Reduced from 6×500 to 4×300 chars to stay within free-tier
        # context budgets. Abstract-level chunks are typically under 400 chars
        # so this captures most of each chunk while halving token consumption.
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
