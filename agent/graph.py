"""LangGraph state machine for agentic RAG routing."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    ask_clarification,
    call_arxiv_tool,
    check_context,
    classify_intent,
    generate_answer,
    refuse,
    rewrite_query_node,
    update_memory,
)


class AgentState(TypedDict, total=False):
    """State tracked across graph execution."""

    messages: list
    query: str
    rewritten_query: str
    retrieved_docs: list
    tool_result: str
    decision: str
    answer: str
    trace: list
    summary: str
    turn_count: int
    context_decision: str
    sources: list
    rewrite_enabled: bool


def _route_intent(state: AgentState) -> str:
    """Return node key from classify_intent output."""
    return state.get("decision", "clarify")


def _route_context(state: AgentState) -> str:
    """Route based on context quality and query specificity."""
    decision = state.get("context_decision", "empty")
    query = state.get("query", "")
    specific = any(token in query.lower() for token in ["latest", "this week", "last month", "by ", "paper"])
    if decision == "sufficient":
        return "generate_answer"
    if decision == "empty" and specific:
        return "call_arxiv_tool"
    return "ask_clarification"


def build_graph():
    """Construct and compile the assignment-specified LangGraph state machine."""
    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("check_context", check_context)
    graph.add_node("call_arxiv_tool", call_arxiv_tool)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("refuse", refuse)
    graph.add_node("update_memory", update_memory)
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "retrieve": "rewrite_query",
            "clarify": "ask_clarification",
            "refuse": "refuse",
            "tool": "call_arxiv_tool",
            "answer_from_memory": "generate_answer",
        },
    )
    graph.add_edge("rewrite_query", "check_context")
    graph.add_conditional_edges(
        "check_context",
        _route_context,
        {
            "generate_answer": "generate_answer",
            "ask_clarification": "ask_clarification",
            "call_arxiv_tool": "call_arxiv_tool",
        },
    )
    graph.add_edge("call_arxiv_tool", "generate_answer")
    graph.add_edge("generate_answer", "update_memory")
    graph.add_edge("update_memory", END)
    graph.add_edge("ask_clarification", END)
    graph.add_edge("refuse", END)
    return graph.compile()
